using LinearAlgebra: ⋅

include("chapter_4.jl")

abstract type DescentMethod end

function iterate!(M::DescentMethod, f, ∇f, x, n)

    for i in 1:n
        x = step!(M::DescentMethod, f, ∇f, x)
    end

    return x

end

## gradient descent

mutable struct GradientDescent <: DescentMethod
    α
end

function init!(M::GradientDescent, f, ∇f, x)

    # do nothing
    return M

end

function step!(M::GradientDescent, f, ∇f, x)

    # compute gradient and access parameters
    g, α = ∇f(x), M.α

    # update and return new x value
    x′ = x - α*g

    return x′

end

# gradient descent example

f(x)  =   x[1]^2 + x[2]^2 + 0.1x[2]^4
∇f(x) = [2x[1],   2x[2]+0.4x[2]^3]
x     = [5, 4]

gd_method = GradientDescent(0.1)
init!(gd_method, f, ∇f, x)
x = iterate!(gd_method, f, ∇f, x, 6)

## conjugate gradient descent

mutable struct ConjugateGradientDescent <: DescentMethod
    d
    g
    update
end

function init!(M::ConjugateGradientDescent, f, ∇f, x)

    # compute the gradient and descent direction
    M.g = ∇f(x)
    M.d = -M.g
    return M

end

function polak_ribiere_update(g, g′)

    return max(0, (g′⋅(g′-g)) / (g⋅g))

end

function fletcher_reeves_update(g, g′)

    return (g′⋅g′) / (g⋅g)

end

function step!(M::ConjugateGradientDescent, f, ∇f, x)

    # access current descent direction and gradient, and compute new gradient
    d, g, g′ = M.d, M.g, ∇f(x)

    # update β using update function, compute new descent direction, and store
    β = M.update(g, g′)
    d′ = -g′ + β*d
    M.d, M.g = d′, g′

    # perform line search in new descent direction and return new x value
    x′ = line_search(f, x, d′)
    return x′

end

# conjugate gradient example

f(x)  =   x[1]^2 + x[2]^2 + 0.1x[2]^4
∇f(x) = [2x[1],   2x[2]+0.4x[2]^3]
x     = [5, 4]

cgd_fr_method = ConjugateGradientDescent(0, 0, fletcher_reeves_update)
init!(cgd_fr_method, f, ∇f, x)
x = iterate!(cgd_fr_method, f, ∇f, x, 6)

cgd_pr_method = ConjugateGradientDescent(0,0,polak_ribiere_update)
init!(cgd_pr_method, f, ∇f, x)
x = iterate!(cgd_pr_method, f, ∇f, x, 6)

## gradient descent with momentum

mutable struct GradientDescentWithMomentum <: DescentMethod
    α # learning rate
    β # momentum decay
    v # momentum
end

function init!(M::GradientDescentWithMomentum, f, ∇f, x)

    # make momentum vector
    M.v = zeros(length(x))
    return M

end

function step!(M::GradientDescentWithMomentum, f, ∇f, x)

    # compute gradient and access parameters
    g, α, β, v = ∇f(x), M.α, M.β, M.v

    # update momentum and return new x value
    v[:] = β*v - α*g
    x′ = x + v

    return x′

end

# gradient descent with momentum example

f(x)  =   x[1]^2 + x[2]^2 + 0.1x[2]^4
∇f(x) = [2x[1],   2x[2]+0.4x[2]^3]
x     = [5, 4]

gdwm_method = GradientDescentWithMomentum(0.1, 0.5, 0)
init!(gdwm_method, f, ∇f, x)
x = iterate!(gdwm_method, f, ∇f, x, 6)


## gradient descent with Nesterov momentum

mutable struct GradientDescentWithNesterovMomentum <: DescentMethod
    α # learning rate
    β # momentum decay
    v # momentum
end

function init!(M::GradientDescentWithNesterovMomentum, f, ∇f, x)

    # make momentum vector
    M.v = zeros(length(x))
    return M

end

function step!(M::GradientDescentWithNesterovMomentum, f, ∇f, x)

    # access parameters
    α, β, v = M.α, M.β, M.v

    # update Nesterov momentum and return new x value
    v[:] = β*v - α*∇f(x + β*v)
    x′ = x + v

    return x′

end

# gradient descent with Nesterov momentum example

f(x)  =   x[1]^2 + x[2]^2 + 0.1x[2]^4
∇f(x) = [2x[1],   2x[2]+0.4x[2]^3]
x     = [5, 4]

gdwnm_method = GradientDescentWithNesterovMomentum(0.1, 0.5, 0)
init!(gdwnm_method, f, ∇f, x)
x = iterate!(gdwnm_method, f, ∇f, x, 6)


## adaptive subgradient (adagrad)

mutable struct AdaptiveSubgradientDescent <: DescentMethod
    α # learning rate
    ϵ # small value
    s # sum of squared gradients
end

function init!(M::AdaptiveSubgradientDescent, f, ∇f, x)

    # make squared gradients vector
    M.s = zeros(length(x))
    return M

end

function step!(M::AdaptiveSubgradientDescent, f, ∇f, x)

    # access parameters and compute gradient
    α, ϵ, s, g = M.α, M.ϵ, M.s, ∇f(x)

    # update sum of squared gradients and return new x value
    s[:] += g.*g
    x′ = x - α*g ./ (sqrt.(s) .+ ϵ)

    return x′

end

# adaptive subgradient descent example

f(x)  =   x[1]^2 + x[2]^2 + 0.1x[2]^4
∇f(x) = [2x[1],   2x[2]+0.4x[2]^3]
x     = [5, 4]

adagrad_method = AdaptiveSubgradientDescent(0.1, 1e-8, 0)
init!(adagrad_method, f, ∇f, x)
x = iterate!(adagrad_method, f, ∇f, x, 6)


## root-mean-square propagation (RMSprop)

mutable struct RootMeanSquarePropagation <: DescentMethod
    α # learning rate
    γ # decay factor
    ϵ # small value
    s # sum of squared gradients
end

function init!(M::RootMeanSquarePropagation, f, ∇f, x)

    # make squared gradients vector
    M.s = zeros(length(x))
    return M

end

function step!(M::RootMeanSquarePropagation, f, ∇f, x)

    # access parameters and compute gradient
    α, γ, ϵ, s, g = M.α, M.γ, M.ϵ, M.s, ∇f(x)

    # update sum of squared gradients and return new x value
    s[:] = γ*s + (1-γ)*(g.*g)
    x′ = x - α*g ./ (sqrt.(s) .+ ϵ)

    return x′

end

# root mean square propagation example

f(x)  =   x[1]^2 + x[2]^2 + 0.1x[2]^4
∇f(x) = [2x[1],   2x[2]+0.4x[2]^3]
x     = [5, 4]

rmsprop_method = RootMeanSquarePropagation(0.1, 0.9, 1e-8, 0)
init!(rmsprop_method, f, ∇f, x)
x = iterate!(rmsprop_method, f, ∇f, x, 6)


## adaptive delta (adadelta)

mutable struct AdaptiveDelta <: DescentMethod
    γs # gradient decay factor
    γx # update decay factor
    ϵ # small value
    s # sum of squared gradients
    u # sum of squared updates
end

function init!(M::AdaptiveDelta, f, ∇f, x)

    # make squared gradients and squared updates vectors
    M.s = zeros(length(x))
    M.u = zeros(length(x))
    return M

end

function step!(M::AdaptiveDelta, f, ∇f, x)

    # access parameters and compute gradient
    γs, γx, ϵ, s, u, g = M.γs, M.γx, M.ϵ, M.s, M.u, ∇f(x)

    # update sum of squared gradients and sum of squared updates
    # and return new x value
    s[:] = γs*s + (1-γs)*(g.*g)
    Δx = - (sqrt.(u) .+ ϵ) ./ (sqrt.(s) .+ ϵ) .* g
    u[:] = γx*u + (1-γx)*(Δx.*Δx)
    x′ = x + Δx

    return x′

end

# adaptive delta example

f(x)  =   x[1]^2 + x[2]^2 + 0.1x[2]^4
∇f(x) = [2x[1],   2x[2]+0.4x[2]^3]
x     = [5, 4]

adadelta_method = AdaptiveDelta(0.999, 0.9, 1e-8, 0, 0)
init!(adadelta_method, f, ∇f, x)
x = iterate!(adadelta_method, f, ∇f, x, 100)


## adaptive moment estimation (adam)

mutable struct AdaptiveMomentEstimation <: DescentMethod
    α # learning rate
    γv # 1st moment decay factor
    γs # 2nd moment decay factor
    ϵ # small value
    k # step counter
    v # 1st moment estimate
    s # 2nd moment estimate
end

function init!(M::AdaptiveMomentEstimation, f, ∇f, x)

    # ensure k=0 and make 1st and 2nd moment vectors
    M.k = 0
    M.v = zeros(length(x))
    M.s = zeros(length(x))
    return M

end

function step!(M::AdaptiveMomentEstimation, f, ∇f, x)

    # access parameters and compute gradient
    α, γv, γs, ϵ, k, v, s, g = M.α, M.γv, M.γs, M.ϵ, M.k, M.v, M.s, ∇f(x)

    # update first and second moment estimates
    v[:] = γv*v + (1-γv)*(g)
    s[:] = γs*s + (1-γs)*(g.*g)
    M.k = k += 1

    # perform bias correction
    v_hat = v ./ (1 - γv^k)
    s_hat = s ./ (1 - γs^k)

    # update and return new x value
    x′ = x - α*v_hat ./ (sqrt.(s_hat) .+ ϵ)

    return x′

end

# adaptive moment estimation example

f(x)  =   x[1]^2 + x[2]^2 + 0.1x[2]^4
∇f(x) = [2x[1],   2x[2]+0.4x[2]^3]
x     = [5, 4]

adam_method = AdaptiveMomentEstimation(0.001, 0.9, 0.999, 1e-8, 0, 0, 0)
init!(adam_method, f, ∇f, x)
x = iterate!(adam_method, f, ∇f, x, 10)


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

# hypergradient descent example

f(x)  =   x[1]^2 + x[2]^2 + 0.1x[2]^4
∇f(x) = [2x[1],   2x[2]+0.4x[2]^3]
x     = [5, 4]

hgd_method = HypergradientDescent(0.001, 1e-4, 0.1, 0)
init!(hgd_method, f, ∇f, x)
x = iterate!(hgd_method, f, ∇f, x, 10)

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
    #v[:] = β*v - α*∇f(x + β*v) # from Nesterov momentum algorithm
    v[:] = β*v + g           # from book
    M.g_prev, M.α = g, α
    #x′ = x + v                  # from Nesterov momentum algorithm
    x′ = x - α*(g + β*v)      # from book

    return x′

end

# hypergradient descent with Nesterov momentum example

f(x)  =   x[1]^2 + x[2]^2 + 0.1x[2]^4
∇f(x) = [2x[1],   2x[2]+0.4x[2]^3]
x     = [5, 4]

hgdwnm_method = HypergradientDescentWithNesterovMomentum(1e-1, 1e-5, 0.5, 0, 0, 0)
init!(hgdwnm_method, f, ∇f, x)
x = iterate!(hgdwnm_method, f, ∇f, x, 6)
