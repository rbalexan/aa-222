using LinearAlgebra
using Statistics
using Distributions

abstract type ConstrainedOptimizationMethod end
abstract type UnconstrainedOptimizationMethod end

include("nesterov_momentum.jl")
include("penalty_method.jl")
include("interior_point_method.jl")



"""
    optimize(f, g, c, x0, n, prob)

Arguments:
    - `f`: Function to be optimized
    - `g`: Gradient function for `f`
    - `c`: Constraint function for 'f'
    - `x0`: (Vector) Initial position to start from
    - `n`: (Int) Number of evaluations allowed. Remember `g` costs twice of `f`
    - `prob`: (String) Name of the problem. So you can use a different strategy for each problem. E.g. "simple1", "secret2", etc.

Returns:
    - The location of the minimum
"""
function optimize(f, g, c, x0, n, prob)

    #if prob == "simple1" || prob == "simple2" || prob == "simple3"
    #    descent_method = Adam(2e-1, 0.7, 0.99, 1e-6, 0, 0, 0)
    #else
    #    descent_method = Adam(3e-1, 0.7, 0.99, 1e-6, 0, 0, 0)
    #end

    #x = x0

    exterior_unconstrained_method = NesterovMomentum(1e-3, 0.9, [])
    exterior_method               = PenaltyMethod(1e0, 1e0, 2.0, quadratic_penalty,
                                        forward_difference_penalty_gradient,
                                        exterior_unconstrained_method, true)
    #interior_unconstrained_method = hooke_jeeves_method
    interior_method               = InteriorPointMethod(1.0, 2.0, 1e-3, Inf,
                                        inverse_barrier)

    history = []
    push!(history, x0)

    optimize!(history, exterior_method, interior_method, f, g, c, x0, n)

    #filter!(x -> all(.!isnan.(x)), history)
    #@show history
    #@show count(f,g,c)
    @show history[end]
    return history[end]

end


function optimize!(history, exterior_method, interior_method, f, g, c, x, n; k=0)

    init!(exterior_method.unconstrained_opt_method, f, g, x)
    feasible = false

    # get a feasible point
    while !feasible && count(f, g, c) < n && k < n

        x, feasible = step!(exterior_method, f, g, c, x)
        push!(history, x)

        k += 1

    end

    #@show x

    # search the feasible space
    while count(f, g, c) < n && k < n

        x = step!(interior_method, f, g, c, x)
        push!(history, x)

        k += 1

    end

    return history

end
