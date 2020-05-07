mutable struct InteriorPointMethod <: ConstrainedOptimizationMethod
    ρ::Float64
    γ::Float64
    ϵ::Float64
    δ::Float64
    barrier::Function
    #unconstrained_opt_method::UnconstrainedOptimizationMethod
end


function step!(M::InteriorPointMethod, f::Function, g::Function, c::Function, x::Vector)

    fb(x) = f(x) + M.barrier(M, c, x)
    x′ = hooke_jeeves_method(fb, x, 1e-1, 1e-3)

    M.δ = norm(x′ - x)
    x   = x′
    M.ρ *= M.γ

    return x′

end


## barrier functions

function inverse_barrier(M::InteriorPointMethod, c::Function, x::Vector)

    constraint_values = c(x)
    barrier_penalty   = any(constraint_values .> 0) ? Inf : -sum(1 ./ constraint_values)

    return 1/M.ρ*barrier_penalty

end


## Hooke-Jeeves method

basis(i, n) = [k == i ? 1.0 : 0.0 for k in 1:n]

function hooke_jeeves_method(f::Function, x::Vector, α::Float64, ϵ::Float64; γ=0.5, k=0, k_max=10)

    # evaluate function and dimensionality
    y, n = f(x), length(x)

    # while step size is greater than tolerance
    while α > ϵ && k < k_max

        # initialize improvement metrics
        improved = false
        x_best, y_best = x, y

        # iterate over pattern
        for i in 1:n, sgn in (-1,1)

            x′ = x + sgn*α*basis(i, n)
            y′ = f(x′)

            if y′ < y_best
                x_best, y_best, improved = x′, y′, true
            end

        end

        # update x and y
        x, y = x_best, y_best

        α *= !improved ? γ : 1

        k += 1

    end

    return x

end
