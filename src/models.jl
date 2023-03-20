import DroneSurveillance: DSState, DSPos
import POMDPTools: SparseCat
import LinearAlgebra: normalize

abstract type DSTransitionModel end
struct DSPerfectModel <: DSTransitionModel
    agent_strategy :: DSAgentStrat
end
struct DSRandomModel <: DSTransitionModel
    uniform_belief
    DSRandomModel(mdp) = new(make_uniform_belief(mdp))
end
struct DSLinModel{T} <: DSTransitionModel where T <: Real
    θ :: AbstractMatrix{T}
    states_buffer :: MVector{21*21+1, DSState}
end
DSLinModel(θ) = DSLinModel{eltype(θ)}(θ, MVector{21*21+1, DSState}(undef))
mutable struct DSLinCalModel{T} <: DSTransitionModel where T <: Real
    lin_model :: DSLinModel{T}
    temperature :: Float64
end
struct DSConformalizedModel{T} <: DSTransitionModel where T <: Real
    lin_model :: DSLinModel{T}
    conf_map :: Dict{Float64, Float64}
end

function predict(mdp, model::DSLinModel, s::DSState, a::DSPos; ϵ_prune=1e-4, T=1.0)
    nx, ny = mdp.size
    states = model.states_buffer
    for (i, (Δx, Δy)) in enumerate(product(-nx:nx, -ny:ny))
        states[i] = Δs_to_s(mdp, s, a, (Δx, Δy))
    end
    states[end] = mdp.terminal_state

    Δx = s.agent.x - s.quad.x
    Δy = s.agent.y - s.quad.y
    ξ = [Δx, Δy, a.x, a.y, 1]
    softmax(x) = exp.(x./T) / sum(exp.(x./T))
    probs = softmax(model.θ * ξ)

    # we prune states with small probability
    return prune_states(SparseCat(states, probs), ϵ_prune)
end

predict(mdp, cal_model::DSLinCalModel, s::DSState, a::DSPos; ϵ_prune=1e-4) =
    predict(mdp, cal_model.lin_model, s, a; ϵ_prune=ϵ_prune, T=cal_model.temperature)

# make a prediction set with the linear model
function predict(mdp, model::Union{DSLinModel, DSLinCalModel}, s::DSState, a::DSPos, λ::Real; ϵ_prune=1e-4)
    distr = predict(mdp, model, s, a; ϵ_prune=ϵ_prune)

    # Shuffle predictions, keep adding to prediction set until just over or just under
    # desired probability (whichever has smaller "gap" to λ).
    pred_set = begin
        perm = shuffle(eachindex(distr.probs))
        p_perm = distr.probs[perm]
        p_cum = cumsum(p_perm)

        idx = begin
            idx = findfirst(>=(λ), p_cum)
            gap_hi = p_cum[idx] - λ
            gap_lo = λ - get(p_cum, idx-1, 0)
            (gap_hi < gap_lo ? idx : idx-1)
        end

        val_perm = distr.vals[perm]
        Set(val_perm[1:idx])
    end
    return pred_set
end

function predict(mdp, conf_model::DSConformalizedModel, s::DSState, a::DSPos, λ::Real; _ϵ_prune=1e-4)
    lin_model = conf_model.lin_model
    states = lin_model.states_buffer
    nx, ny = mdp.size
    # push!(Δ_states, -2 .* mdp.size)  # this is the "code" for moving to the terminal state
    for (i, (Δx, Δy)) in enumerate(product(-nx:nx, -ny:ny))
        states[i] = Δs_to_s(mdp, s, a, (Δx, Δy))
    end
    states[end] = mdp.terminal_state

    Δx = s.agent.x - s.quad.x
    Δy = s.agent.y - s.quad.y
    ξ = [Δx, Δy, a.x, a.y, 1]
    softmax(x) = exp.(x) / sum(exp.(x))
    probs = softmax(lin_model.θ * ξ)
    λ_hat = conf_model.conf_map[λ]

    idx = probs .>= (1-λ_hat)
    pred_set = states[idx] |> Set
    return pred_set
end

function prune_states(sc::SparseCat, ϵ_prune)
    idx = sc.probs .>= ϵ_prune
    SparseCat(sc.vals[idx], normalize(sc.probs[idx], 1))
end

# this function turns out to slow us down quite a bit...
function project_inbounds(mdp, s::DSState)
    nx, ny = mdp.size
    @assert nx == ny
    if s.quad.x ∈ 1:nx &&
       s.quad.y ∈ 1:ny &&
       s.agent.x ∈ 1:nx &&
       s.agent.y ∈ 1:ny
        return s
    else
        return DSState(clamp.(s.quad, 1, nx), clamp.(s.agent, 1, nx))
    end
end

function Δs_to_s(mdp, s, a, (Δx, Δy)::Tuple)::DSState
    if (Δx, Δy) != -2 .* mdp.size
        s_ = DSState((s.quad + a), s.quad + a + [Δx, Δy])
        project_inbounds(mdp, s_)
    else
        mdp.terminal_state
    end
end
