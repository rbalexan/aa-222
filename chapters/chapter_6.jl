using LinearAlgebra: norm, I, ⋅

include("chapter_4.jl") # for line_search
include("chapter_5.jl") # for iterate!

abstract type DescentMethod end

## newton's method

function newtons_method(∇f, ∇²f, x, ϵ, k_max)

    # initialize k and Δ
    k, Δ = 1, fill(Inf, length(x))

    # iterate until termination
    while k ≤ k_max && norm(Δ) > ϵ

        # compute the update and iterate
        Δ = ∇²f(x) \ ∇f(x)
        x -= Δ
        k += 1

    end

    return x, k

end

# newton's method example

f(x)   = (x[1]+2x[2]-7)^2 + (2x[1]+x[2]-5)^2 + x[1]^4
∇f(x)  = [10x[1]+8x[2]-34+4x[1]^3, 8x[1]+10x[2]-38]
∇²f(x) = [10+12x[1]^2    8
          8              10]

x = [9, 8]
x, k = newtons_method(∇f, ∇²f, x, 1e-6, 20)

## secant method

function secant_method(f′, x0, x1, ϵ, k_max)

    # initialize k and Δ and compute the initial gradient
    k, Δ = 1, Inf
    g0 = f′(x0)

    # iterate until termination
    while k ≤ k_max && abs(Δ) > ϵ

        # compute the update and iterate
        g1 = f′(x1)
        Δ = (x1 - x0) / (g1 - g0) * g1
        x0, x1, g0 = x1, x1-Δ, g1
        k += 1

    end

    return x1, k

end

# secant method example

f(x) = 5 + x + 5x^2 + x^4
f′(x) = 1 + 10x + 4x^3

x0, x1 = 9, 8
x, k = secant_method(f′, x0, x1, 1e-6, 20)

### Davidon-Fletcher-Powell (DFP) method

mutable struct DFP <: DescentMethod
    Q # approximate inverse Hessian
end

function init!(M::DFP, f, ∇f, x)

    # initialize Q
    m = length(x)
    M.Q = Matrix(1.0I, m, m)

    return M

end

function step!(M::DFP, f, ∇f, x)

    # access Q and compute the gradient
    Q, g = M.Q, ∇f(x)

    # compute new iterate and gradient
    x′ = line_search(f, x, -Q*g)
    g′ = ∇f(x′)

    # compute DFP Q-update
    δ, γ = x′ - x, g′ - g
    Q[:] = Q - Q*γ*γ'*Q/(γ'*Q*γ) + δ*δ'/(δ'*γ)

    return x′

end

# DFP method example

f(x)   = (x[1]+2x[2]-7)^2 + (2x[1]+x[2]-5)^2 + x[1]^4
∇f(x)  = [10x[1]+8x[2]-34+4x[1]^3, 8x[1]+10x[2]-38]

x0 = [9, 8]

dfp_method = DFP(0)
init!(dfp_method, f, ∇f, x0)
x = iterate!(dfp_method, f, ∇f, x0, 6)

## Broyden-Fletcher-Goldfarb-Shanno (BFGS) method

mutable struct BFGS <: DescentMethod
    Q # approximate inverse Hessian
end

function init!(M::BFGS, f, ∇f, x)

    # initialize Q
    m = length(x)
    M.Q = Matrix(1.0I, m, m)

    return M

end

function step!(M::BFGS, f, ∇f, x)

    # access Q and compute the gradient
    Q, g = M.Q, ∇f(x)

    # compute new iterate and gradient
    x′ = line_search(f, x, -Q*g)
    g′ = ∇f(x′)

    # compute BFGS Q-update
    δ, γ = x′ - x, g′ - g
    Q[:] = Q - (δ*γ'*Q + Q*γ*δ')/(δ'*γ) + (1 + (γ'*Q*γ)/(δ'*γ))[1]*(δ*δ')/(δ'*γ)

    return x′

end

# BFGS method example

f(x)   = (x[1]+2x[2]-7)^2 + (2x[1]+x[2]-5)^2 + x[1]^4
∇f(x)  = [10x[1]+8x[2]-34+4x[1]^3, 8x[1]+10x[2]-38]

x0 = [9, 8]

bfgs_method = BFGS(0)
init!(bfgs_method, f, ∇f, x0)
x = iterate!(bfgs_method, f, ∇f, x0, 6)

## limited-memory Broyden-Fletcher-Goldfarb-Shanno (L-BFGS) method

mutable struct LimitedMemoryBFGS <: DescentMethod
    m # history parameter
    δs # iterate difference vector
    γs # gradient difference vector
    qs # approximation vector
end

function init!(M::LimitedMemoryBFGS, f, ∇f, x)

    # initialize vectors
    M.δs = []
    M.γs = []
    M.qs = []

    return M

end

function step!(M::LimitedMemoryBFGS, f, ∇f, x)

    # access parameters and compute the gradient
    δs, γs, qs, g = M.δs, M.γs, M.qs, ∇f(x)
    m = length(δs)


    if m > 0 # mth-order LBFGS update

        # compute q vectors
        q = g

        for i in m:-1:1 # traverse backwards to compute qs using L-BFGS update
            qs[i] = copy(q)
            q -= (δs[i]⋅q)/(γs[i]⋅δs[i])*γs[i]
        end

        # compute z vectors
        z = (γs[m].*δs[m].*q)/(γs[m]⋅γs[m])
        for i in 1:m # traverse forwards to compute zs using L-BFGS update
            z += δs[i]*(δs[i]⋅qs[i] - γs[i]⋅z)/(γs[i]⋅δs[i])
        end

        x′ = line_search(f, x, -z)

    else # zero-order LBFGS update

        x′ = line_search(f, x, -g)

    end

    # compute new gradient
    g′ = ∇f(x′)

    # push δ and γ to δs and γs, clear q, and remove old vectors
    push!(δs, x′ - x)
    push!(γs, g′ - g)
    push!(qs, zeros(length(x)))

    while length(δs) > M.m
        popfirst!(δs); popfirst!(γs); popfirst!(qs)
    end

    return x′

end

# limited-memory BFGS method example

f(x)   = (x[1]+2x[2]-7)^2 + (2x[1]+x[2]-5)^2 + x[1]^4
∇f(x)  = [10x[1]+8x[2]-34+4x[1]^3, 8x[1]+10x[2]-38]

x0 = [9, 8]

lbfgs_method = LimitedMemoryBFGS(2, 0, 0, 0)
init!(lbfgs_method, f, ∇f, x0)
x = iterate!(lbfgs_method, f, ∇f, x0, 6)
