using LinearAlgebra
using Statistics

abstract type DescentMethod end

include("gradient_descent_with_nesterov_momentum.jl")
include("adam.jl")
include("hypergradient_descent_with_nesterov_momentum.jl")

"""
    optimize(f, g, x0, n, prob)

Arguments:
    - `f`: Function to be optimized
    - `g`: Gradient function for `f`
    - `x0`: (Vector) Initial position to start from
    - `n`: (Int) Number of evaluations allowed. Remember `g` costs twice of `f`
    - `prob`: (String) Name of the problem. So you can use a different strategy for each problem. E.g. "simple1", "secret2", etc.

Returns:
    - The location of the minimum
"""
function optimize(f, g, x0, n, prob)

    if prob == "simple1" || prob == "simple2" || prob == "simple3"
        descent_method = Adam(2e-1, 0.7, 0.99, 1e-6, 0, 0, 0)
    else
        descent_method = Adam(3e-1, 0.7, 0.99, 1e-6, 0, 0, 0)
    end

    x = x0
    history = optimize_w_history(descent_method, f, g, x0, n)

    filter!(x -> all(.!isnan.(x)), history)
    return history[end]

end

function optimize_w_history(descent_method, f, g, x0, n)

    x = x0
    init!(descent_method, f, g, x)

    history = []
    push!(history,x0)

    k = 0

    while count(f, g) < n && k < n
        x = step!(descent_method, f, g, x)
        push!(history, x)
        k+=1
    end

    return history

end
