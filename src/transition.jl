import Base: product
import Base
import LinearAlgebra: normalize, normalize!
import DroneSurveillance: DSTransitionModel, DSPerfectModel, DSLinModel, DSLinCalModel, DSConformalizedModel

Base.:*(λ::Real, d::Deterministic) = SparseCat([d.val], [λ])
Base.:*(λ::Real, sc::SparseCat) = SparseCat(sc.vals, λ.*sc.probs)
"Add SparseCats (usually have to be multiplied by weight first)."
function ⊕(sc_lhs::SparseCat, sc_rhs::SparseCat)
    SparseCat(vcat(sc_lhs.vals, sc_rhs.vals),
              vcat(sc_lhs.probs, sc_rhs.probs))
end
"Multiply SparseCat"
⊗(d_lhs::Deterministic, sc_rhs::SparseCat) = SparseCat([d_lhs.val], [1]) ⊗ sc_rhs
⊗(sc_lhs::SparseCat, d_rhs::Deterministic) =  sc_lhs ⊗ SparseCat([d_rhs.val], [1])
function ⊗(sc_lhs::SparseCat, sc_rhs::SparseCat)
    vals = product(sc_lhs.vals, sc_rhs.vals) |> collect
    probs = map(prod, product(sc_lhs.probs, sc_rhs.probs)) |> collect
    return SparseCat(vals[:], probs[:])
end

function POMDPs.transition(mdp::DroneSurveillanceMDP, s::DSState, a::DSPos) :: Union{Deterministic, SparseCat}
    T_model = DSPerfectModel(mdp.agent_strategy)
    return transition(mdp, T_model, s, a)
end

# for perfect model
function transition(mdp::DroneSurveillanceMDP, transition_model::DSPerfectModel, s::DSState, a::DSPos; ϵ_prune=1e-4) :: Union{Deterministic, SparseCat}
    agent_strategy = transition_model.agent_strategy
    if isterminal(mdp, s) || s.quad == s.agent || s.quad == mdp.region_B
        return Deterministic(mdp.terminal_state) # the function is not type stable, returns either Deterministic or SparseCat
    else
        # first, move quad
        # if it would move out of bounds, just stay in place
        actor_inbounds(actor_state) = (0 < actor_state[1] <= mdp.size[1]) && (0 < actor_state[2] <= mdp.size[2])
        new_quad = actor_inbounds(s.quad + a) ? s.quad + a : s.quad
        new_quad_distr = SparseCat([new_quad, s.quad], [3//4, 1//4])

        # then, move agent (independently)
        new_agent_distr = move_agent(mdp, agent_strategy, new_quad, s)

        # combine probability distributions of quad and agent
        new_state_dist = let new_state_distr = new_quad_distr ⊗ new_agent_distr
            states = [DSState(q, a) for (q, a) in new_state_distr.vals]
            SparseCat(states, new_state_distr.probs)
        end
        return prune_states(new_state_dist, ϵ_prune)
    end
end

# for linear and linear calibrated model
function transition(mdp::DroneSurveillanceMDP, transition_model::Union{DSLinModel, DSLinCalModel}, s::DSState, a::DSPos; ϵ_prune=1e-4) :: Union{Deterministic, SparseCat}
    if isterminal(mdp, s) || s.quad == s.agent || s.quad == mdp.region_B
        return Deterministic(mdp.terminal_state) # the function is not type stable, returns either Deterministic or SparseCat
    else
        predict(mdp, transition_model, s, a; ϵ_prune=ϵ_prune)
    end
end

# for conformalized model
function transition(mdp::DroneSurveillanceMDP, transition_model::DSConformalizedModel, s::DSState, a::DSPos; ϵ_prune=1e-4)::Dict{<:Real, Set{DSState}}
    if isterminal(mdp, s) || s.quad == s.agent || s.quad == mdp.region_B
        return Dict(
            λ => Set([mdp.terminal_state])
            for λ in keys(transition_model.conf_map)
        )
    else
        return Dict(
            λ => predict(mdp, transition_model, s, a, λ)
            for λ in keys(transition_model.conf_map))
    end
end

function transition(mdp::DroneSurveillanceMDP, transition_model::DSRandomModel, s::DSState, a::DSPos; ϵ_prune=1e-4)
    b0 = transition_model.uniform_belief
    idx = rand(eachindex(b0.probs), 10)
    SparseCat(b0.vals[idx], normalize(b0.probs[idx], 1))
end


function agent_optimal_action_idx(s::DSState) :: Int
    vec = s.quad - s.agent
    # similar to dot = atan2(vec'*[1; 0])
    angle = atan(vec[2], vec[1])
    if π/4 <= angle < π*3/4
        a = ACTIONS_DICT[:north]
    elseif -π/4 <= angle <= π/4
        a = ACTIONS_DICT[:east]
    elseif -π*3/4 <= angle < -π/4
        a = ACTIONS_DICT[:south]
    else 
        a = ACTIONS_DICT[:west]
    end
    return a
end

function move_agent(mdp::DroneSurveillanceMDP, agent_strategy::DSAgentStrat, new_quad::DSPos, s::DSState)
    entity_inbounds(entity_state) = (0 < entity_state[1] <= mdp.size[1]) && (0 < entity_state[2] <= mdp.size[2])
    @assert entity_inbounds(s.agent) "Tried to move agent that's already out of bounds! $(s.agent), $(mdp.size)"

    perfect_agent = begin
        act_idx = agent_optimal_action_idx(s)
        act = ACTION_DIRS[act_idx]
        new_agent = entity_inbounds(s.agent + act) ? s.agent + act : s.agent
        @assert entity_inbounds(new_agent) "Somehow the new agent is out of bounds??"
        Deterministic(new_agent)
    end
    random_agent = begin
        new_agent_states = MVector{N_ACTIONS, DSPos}(undef)
        probs = @MVector(zeros(N_ACTIONS))
        for (i, act) in enumerate(ACTION_DIRS)
            new_agent = entity_inbounds(s.agent + act) ? s.agent + act : s.agent
            if entity_inbounds(new_agent)
                new_agent_states[i] = new_agent
                # Add extra probability to action in direction of drone
                # just go randomly
                probs[i] += 1.0
            else
                @assert false "We should never get here. Maybe the agent was initialized out of bounds in the first place?"
            end
        end
        normalize!(probs, 1)
        SparseCat(new_agent_states, probs)
    end
    return (agent_strategy.p*perfect_agent) ⊕ ((1-agent_strategy.p)*random_agent)
end

function make_uniform_belief(mdp::DroneSurveillanceMDP)
    nx, ny = mdp.size
    b0 = begin
        states = []
        for ax in 1:nx,
            ay in 1:ny,
            dx in 1:nx,
            dy in 1:ny

            if [dx, dy] != [ax ay] && [dx, dy] != mdp.region_B
                push!(states, DSState([dx, dy], [ax, ay]))
            end
        end
        probs = normalize!(ones(length(states)), 1)
        SparseCat(states, probs)
    end
    return b0
end
