using Base.MathConstants: φ
using LinearAlgebra: ⋅
using Convex, SCS

include("chapter_3.jl")

## line search

"""
    line_search(f, x, d)
Perform a one-dimensional line search on `f` at `x` in direction `d`.
"""
function line_search(f, x, d; n=10)

    # define 1-d objective along d and solve
    objective = α -> f(x + α*d)
    a, b = bracket_minimum(objective)

    # assuming lower bound is valid minimizer
    (αl, αu) = golden_section_search(objective, a, b, n) # any minimization algotihm: minimize(objective, a, b)

    return x + αl*d

end

# line search example
#f(x) = x[1]^2 + x[2]^2
#x = [5., 3.]
#d = [-1., -0.7]
#x = line_search(f, x, d)


## backtracking line search

"""
    backtracking_line_search(f, ∇f, x, d, α)
Perform a one-dimensional backtracking line search on `f` at `x` in direction `d`
with initial step `α` utilizing the gradient `∇f`.
"""
function backtracking_line_search(f, ∇f, x, d, α; p=0.5, β=1e-4)

    # compute function and gradient evaluations
    y, g = f(x), ∇f(x)

    # iteratively backtrack until sufficient-decrease condition is satisfied
    while f(x + α*d) > y + β*α*(g⋅d)
        α *= p
    end

    return x + α*d

end

# backtracking line search example
#f(x)  =   x[1]^2 +  x[2]^2
#∇f(x) = [2x[1],    2x[2]]
#x = [5., 3.]
#d = [-1., -0.7]
#x = backtracking_line_search(f, ∇f, x, d, 10, β=0.5)


## strong backtracking line search

"""
    strong_backtracking_line_search(f, ∇f, x, d)
Perform a one-dimensional strong backtracking line search on `f` at `x` in
direction `d` utilizing the gradient `∇f`.
"""
function strong_backtracking_line_search(f, ∇f, x, d; α=0.1, β=1e-4, σ=0.1)

    # compute function and gradient evaluations an initialize y and α
    y0, g0, y_prev, α_prev = f(x), ∇f(x)⋅d, NaN, 0
    α_lo, α_hi = NaN, NaN

    # bracket phase
    while true

        y = f(x + α*d)

        if y > y0 + β*α*g0 || (!isnan(y_prev) && y ≥ y_prev) # eq 4.7 and 4.8 are satisfied
            α_lo, α_hi = α_prev, α
            break
        end

        g = ∇f(x + α*d)⋅d
        if abs(g) ≤ -σ*g0 # strong Wolfe condition already satisfied
            return x + α*d
        elseif g ≥ 0 # eq 4.9 is satisfied
            α_lo, α_hi = α, α_prev
            break
        end

        # update if no suitable α region found yet
        y_prev, α_prev, α = y, α, 2α

    end

    # zoom phase

    y_lo = f(x + α_lo*d)

    # perform bisection by updating lower bound
    while true

        α = (α_lo + α_hi)/2
        y = f(x + α*d)

        if y > y0 + β*α*g0 || y ≥ y_lo  # eq 4.7 and 4.8 are satisfied
            α_hi = α
        else
            g = ∇f(x + α*d)⋅d
            if abs(g) ≤ -σ*g0 # strong Wolfe condition already satisfied
                return x + α*d
            elseif g*(α_hi - α_lo) ≥ 0 # eq 4.9 is satisfied
                α_hi = α_lo
            end

            # update lower bound if no suitable α region found yet
            α_lo = α

        end

    end

end

# backtracking line search example
#f(x)  =   x[1]^2 +  x[2]^2
#∇f(x) = [2x[1],    2x[2]]
#d = [-1., -0.7]
#x = [5., 3.]
#x = strong_backtracking_line_search(f, ∇f, x, d)


## trust region descent

"""
    solve_trust_region_subproblem(∇f, ∇²f, x0, δ)
Solve the trust region maximum-step subproblem with gradient `∇f`, Hessian `∇²f`,
initial point `x0` and trust region radius `δ`.
"""
function solve_trust_region_subproblem(∇f, ∇²f, x0, δ)

    # frame and solve the maximum step problem using Convex package
    x = Variable(length(x0))
    p = minimize(∇f(x0)⋅(x-x0) + quadform(x-x0, ∇²f(x0))/2)
    p.constraints += norm(x-x0) <= δ
    solve!(p, SCS.Optimizer(verbose=false))

    return (x.value, p.optval)

end

"""
    trust_region_descent(f, ∇f, ∇²f, x, k_max)
Perform at most `k_max` iterations of trust region descent on function `f`
starting at initial point `x0` with gradient `∇f` and Hessian `∇²f`.
"""
function trust_region_descent(f, ∇f, ∇²f, x, k_max;
    η1=0.25, η2=0.5, γ1=0.5, γ2=2.0, δ=1.0)

    y = f(x)

    for k in 1:k_max

        x′, y′ = solve_trust_region_subproblem(∇f, ∇²f, x, δ)

        # compute improvement ratio
        η = (y - f(x′)) / (y - y′)

        # expand or contract trust region based on expected improvement
        δ   *= η < η1 ? γ1 : 1
        δ   *= η > η2 ? γ2 : 1

        # if reasonable improvement, update values
        x, y = η ≥ η1 ? (x′, f(x)) : (x, y)
        @show x, y, δ, η

    end

    return x

end

# trust region descent example
#f(x)   =   x[1]^2 + x[2]^2 + x[1]^4 + x[2]^4
#∇f(x)  = [2x[1] + 4x[1]^3, 2x[2] + 4x[2]^3]
#∇²f(x) = [2+12x[1]^2 0;
#          0 2+12x[2]^2]
#∇²f_error(x) = [2+11x[1]^2 0;
#                0 2+11x[2]^2]
#x = [-4., 10.]
#x = trust_region_descent(f, ∇f, ∇²f, x, 15)

#x = [-4., 10.]
#x = trust_region_descent(f, ∇f, ∇²f_error, x, 15)
