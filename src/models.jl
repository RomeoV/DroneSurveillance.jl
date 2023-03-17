import DroneSurveillance: DSState, DSPos
import POMDPTools: SparseCat
import LinearAlgebra: normalize

abstract type DSTransitionModel end
struct DSPerfectModel <: DSTransitionModel end
struct DSApproximateModel <: DSTransitionModel end
struct DSLinModel{T} <: DSTransitionModel where T <: Real
    θ_Δx :: AbstractMatrix{T}
    θ_Δy :: AbstractMatrix{T}
end
mutable struct DSLinCalModel{T} <: DSTransitionModel where T <: Real
    lin_model :: DSLinModel{T}
    temperature :: Float64
end
struct DSConformalizedModel{T} <: DSTransitionModel where T <: Real
    lin_model :: DSLinModel{T}
    conf_map_Δx :: Dict{Float64, Float64}
    conf_map_Δy :: Dict{Float64, Float64}
end

function prune_states(sc::SparseCat, ϵ_prune)
    idx = sc.probs .>= ϵ_prune
    SparseCat(sc.vals[idx], normalize(sc.probs[idx], 1))
end

function predict(model::DSLinModel, s::DSState, a::DSPos; ϵ_prune=1e-4, T=1.0)
    nx, ny = size.([model.θ_Δx, model.θ_Δy], 1) .÷ 2
    states_Δx, states_Δy = (-nx:nx, -ny:ny) .|> collect

    Δx = s.agent.x - s.quad.x
    Δy = s.agent.y - s.quad.y
    ξ = [Δx, Δy, a.x, a.y, 1]
    softmax(x) = exp.(x./T) / sum(exp.(x./T))
    probs_Δx, probs_Δy = (softmax(model.θ_Δx * ξ),
                          softmax(model.θ_Δy * ξ))

    # we prune states with small probability
    return (prune_states(SparseCat(states_Δx, probs_Δx), ϵ_prune),
            prune_states(SparseCat(states_Δy, probs_Δy), ϵ_prune))
end

predict(cal_model::DSLinCalModel, s::DSState, a::DSPos; ϵ_prune=1e-4) =
    predict(cal_model.lin_model, s, a; ϵ_prune=ϵ_prune, T=cal_model.T)

# make a prediction set with the linear model
function predict(model::Union{DSLinModel, DSLinCalModel}, s::DSState, a::DSPos, λ::Real; ϵ_prune=1e-4)
    lhs_distr, rhs_distr = predict(model, s, a; ϵ_prune=ϵ_prune)

    # Shuffle predictions, keep adding to prediction set until just over or just under
    # desired probability (whichever has smaller "gap" to λ).
    lhs_pred_set, rhs_pred_set = Tuple([begin
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
        for distr in [lhs_distr, rhs_distr]
    ])
    return lhs_pred_set, rhs_pred_set
end

function predict(conf_model::DSConformalizedModel, s::DSState, a::DSPos, λ::Real; ϵ_prune=1e-4)
    lin_model = conf_model.lin_model
    nx, ny = size.([lin_model.θ_Δx, lin_model.θ_Δy], 1) .÷ 2
    states = (-nx:nx, -ny:ny) .|> collect

    Δx = s.agent.x - s.quad.x
    Δy = s.agent.y - s.quad.y
    ξ = [Δx, Δy, a.x, a.y, 1]
    softmax(x) = exp.(x) / sum(exp.(x))
    probs = (softmax(lin_model.θ_Δx * ξ), softmax(lin_model.θ_Δy * ξ))
    λ_hat_Δx = conf_model.conf_map_Δx[λ]
    λ_hat_Δy = conf_model.conf_map_Δy[λ]

    idx_Δx = probs[1] .>= (1-λ_hat_Δx)
    idx_Δy = probs[2] .>= (1-λ_hat_Δy)
    pred_set_Δx = states[1][idx_Δx] |> Set
    pred_set_Δy = states[2][idx_Δy] |> Set
    return (pred_set_Δx, pred_set_Δy)
end
