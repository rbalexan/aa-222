using LinearAlgebra

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

    exterior_unconstrained_method = NesterovMomentum(1e-3, 0.9, [])
    exterior_method               = PenaltyMethod(1e0, 1e0, 2.0,
                                        quadratic_penalty,
                                        forward_difference_penalty_gradient,
                                        exterior_unconstrained_method, true)
    interior_method               = InteriorPointMethod(1.0, 2.0, inverse_barrier)

    # secret1 has guaranteed feasible start

    history = []
    push!(history, x0)

    init_feasible = prob == "secret1" ? true : false
    # can do n = Inf for plotting
    #n             = prob == "secret2" ? n-200 : n-(4*length(x0)+1)

    optimize!(history, exterior_method, interior_method, f, g, c, x0, n, feasible=init_feasible)

    filter!(x -> all(.!isnan.(x)), history)

    return history
    #return history[end]

end


function optimize!(history, exterior_method, interior_method, f, g, c, x, n; feasible=false)

    init!(exterior_method.unconstrained_opt_method, f, g, x)

    # obtain a feasible point
    while !feasible && count(f, g, c) < n

        x, feasible = step!(exterior_method, f, g, c, x)
        push!(history, x)

    end

    # search the feasible space
    while count(f, g, c) < n # min eval per hooke-jeeves loop plus a few since it doesnt seem to work

        x, feasible = step!(interior_method, f, g, c, x, n)
        push!(history, x)

    end

    return history

end
