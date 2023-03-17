import DroneSurveillance: DSState, DSPos
import POMDPTools: SparseCat
import LinearAlgebra: normalize

abstract type DSTransitionModel end
struct DSPerfectModel <: DSTransitionModel end
struct DSLinModel{T} <: DSTransitionModel where T <: Real
    θ :: AbstractMatrix{T}
    size :: Tuple{Int, Int}
end
mutable struct DSLinCalModel{T} <: DSTransitionModel where T <: Real
    lin_model :: DSLinModel{T}
    temperature :: Float64
end
struct DSConformalizedModel{T} <: DSTransitionModel where T <: Real
    lin_model :: DSLinModel{T}
    conf_map :: Dict{Float64, Float64}
end

function prune_states(sc::SparseCat, ϵ_prune)
    idx = sc.probs .>= ϵ_prune
    SparseCat(sc.vals[idx], normalize(sc.probs[idx], 1))
end

function predict(model::DSLinModel, s::DSState, a::DSPos; ϵ_prune=1e-4, T=1.0)
    nx, ny = model.size
    states = [(Δx, Δy) for Δx in -nx:nx,
                           Δy in -ny:ny][:]

    Δx = s.agent.x - s.quad.x
    Δy = s.agent.y - s.quad.y
    ξ = [Δx, Δy, a.x, a.y, 1]
    softmax(x) = exp.(x./T) / sum(exp.(x./T))
    probs = softmax(model.θ * ξ)

    # we prune states with small probability
    return prune_states(SparseCat(states, probs), ϵ_prune)
end

predict(cal_model::DSLinCalModel, s::DSState, a::DSPos; ϵ_prune=1e-4) =
    predict(cal_model.lin_model, s, a; ϵ_prune=ϵ_prune, T=cal_model.T)

# make a prediction set with the linear model
function predict(model::Union{DSLinModel, DSLinCalModel}, s::DSState, a::DSPos, λ::Real; ϵ_prune=1e-4)
    distr = predict(model, s, a; ϵ_prune=ϵ_prune)

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

function predict(conf_model::DSConformalizedModel, s::DSState, a::DSPos, λ::Real; ϵ_prune=1e-4)
    lin_model = conf_model.lin_model
    nx, ny = model.size
    states = [(Δx, Δy) for Δx in -nx:nx,
                           Δy in -ny:ny][:]

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
