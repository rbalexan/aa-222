## hypergradient descent with Nesterov momentum

mutable struct HypergradientDescentWithNesterovMomentum <: DescentMethod
    α0 # initial learning rate
    μ # learning rate of learning rate
    β # momentum decay
    v # momentum
    α # learning rate
    g_prev # previous gradient
end

function init!(M::HypergradientDescentWithNesterovMomentum, f, ∇f, x)

    # initialize learning rate and momentum and gradient vectors
    M.α = M.α0
    M.v = zeros(length(x))
    M.g_prev = zeros(length(x))
    return M

end

function step!(M::HypergradientDescentWithNesterovMomentum, f, ∇f, x)

    # access parameters
    α, β, μ, v, g, g_prev = M.α, M.β, M.μ, M.v, ∇f(x), M.g_prev

    # update learning rate, Nesterov momentum, store values, and return new x value
    α = α - μ*(g⋅(-g_prev - β*v))
    v[:] = β*v + g
    M.g_prev, M.α = g, α
    x′ = x - α*(g + β*v)

    return x′

end
