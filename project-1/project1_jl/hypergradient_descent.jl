## hypergradient descent

mutable struct HypergradientDescent <: DescentMethod
    α0 # initial learning rate
    μ # learning rate of learning rate
    α # current learning rate
    g_prev # previous gradient
end

function init!(M::HypergradientDescent, f, ∇f, x)

    # initialize learning rate and gradient vector
    M.α = M.α0
    M.g_prev = zeros(length(x))
    return M

end

function step!(M::HypergradientDescent, f, ∇f, x)

    # access parameters and compute gradient
    α, μ, g, g_prev = M.α, M.μ, ∇f(x), M.g_prev

    # update learning rate, store values, and return new x value
    α = α + μ*(g⋅g_prev)
    M.g_prev, M.α = g, α
    x′ = x - α*g

    return x′

end
