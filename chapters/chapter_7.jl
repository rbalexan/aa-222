using Statistics
using Plots; pyplot()
using DataStructures

include("chapter_4.jl") # for line_search

## cyclic coordinate descent

basis(i, n) = [k == i ? 1.0 : 0.0 for k in 1:n]

function cyclic_coordinate_descent(f, x, ϵ)

    # initalize Δ, dimensionality, and counter
    Δ, n, k, x′ = Inf, length(x), 0, x

    # iterate until cycle tolerance is met
    while abs(Δ) > ϵ

        x′ = x

        # perform cyclic coordinate descent
        for i in 1:n

            d = basis(i, n)
            x′ = line_search(f, x′, d)

        end

        Δ = norm(x′ - x)
        k += 1
        x = x′

    end

    return x′, k

end

# cyclic coordinate descent example

f(x) = (x[1]+2x[2]-7)^2 + (2x[1]+x[2]-5)^2
x    = [9, 8]
x, k = cyclic_coordinate_descent(f, x, 1e-3)


## accelerated cyclic coordinate descent

function accelerated_cyclic_coordinate_descent(f, x, ϵ)

    # initalize Δ, dimensionality, and counter
    Δ, n, k, x′ = Inf, length(x), 0, x

    # iterate until cycle tolerance is met
    while abs(Δ) > ϵ

        x′ = x

        # perform cyclic coordinate descent
        for i in 1:n

            d = basis(i, n)
            x′ = line_search(f, x′, d)

        end

        # perform acceleration step
        x′ = line_search(f, x′, x′ - x)

        Δ = norm(x′ - x)
        k += 1
        x = x′

    end

    return x′, k

end

# accelerated cyclic coordinate descent example

f(x) = (x[1]+2x[2]-7)^2 + (2x[1]+x[2]-5)^2
x    = [9, 8]
x, k = accelerated_cyclic_coordinate_descent(f, x, 1e-3)


## Powell's method

function powells_method(f, x, ϵ)

    # initalize Δ, dimensionality, counter, and initial basis vectors
    Δ, n, k, x′ = Inf, length(x), 0, x
    U = [basis(i, n) for i in 1:n]

    # iterate until cycle tolerance is met
    while abs(Δ) > ϵ

        x′ = x

        # perform cyclic coordinate descent
        for i in 1:n

            d = U[i]
            x′ = line_search(f, x′, d)

        end

        # update descent directions
        for i in 1:n-1
            U[i] = U[i+1] # could use deletefirst!() or deleteat!()
        end

        # search along cycle direction
        U[n] = d = x′ - x
        x′ = line_search(f, x′, x′ - x)

        Δ = norm(x′ - x)
        k += 1
        x = x′

    end

    return x′, k

end

# Powell's method example

f(x) = (x[1]+2x[2]-7)^2 + (2x[1]+x[2]-5)^2
x    = [9, 8]
x, k = powells_method(f, x, 1e-3)


## Hooke-Jeeves method

function hooke_jeeves_method(f, x, α, ϵ, γ=0.5)

    # evaluate function and dimensionality
    y, n = f(x), length(x)

    # while step size is greater than tolerance
    while α > ϵ

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

    end

    return x

end

# Hooke-Jeeves method example
f(x) = (x[1]+2x[2]-7)^2 + (2x[1]+x[2]-5)^2
x    = [9, 8]
x, k = hooke_jeeves_method(f, x, 0.4, 1e-3)

## generalized pattern search

function generalized_pattern_search(f, x, α, D, ϵ, γ=0.5)

    # evaluate function and dimensionality
    y, n = f(x), length(x)

    # while step size is greater than tolerance
    while α > ϵ

        # initialize improvement metrics
        improved = false

        # search opportunistically along directions
        for (i, d) in enumerate(D)

            x′ = x + α*d
            y′ = f(x′)

            if y′ < y

                # update x and y
                x, y, improved = x′, y′, true
                D = pushfirst!(deleteat!(D, i), d) # dynamic ordering
                break

            end

        end

        α *= !improved ? γ : 1

    end

    return x

end

# generalized pattern search example
f(x) = (x[1]+2x[2]-7)^2 + (2x[1]+x[2]-5)^2
x    = [9, 8]
D    = [[1, 0], [0, 1], [-1, -1]]
x    = generalized_pattern_search(f, x, 0.4, D, 1e-3)


## Nelder-Mead simplex method

function nelder_mead_simplex_method(f, S, ϵ; α=1.0, β=2.0, γ=0.5)

    # initialize Δ and query simplex points
    Δ, y_arr = Inf, f.(S)
    S_history = [S,]

    # while update is greater than tolerance
    while Δ > ϵ

        # sort and assign simplex values
        p = sortperm(y_arr)             # sort lowest to highest
        S, y_arr = S[p], y_arr[p]       # get sorted simplex and y values
        xl, yl = S[1],     y_arr[1]     # lowest
        xs, ys = S[end-1], y_arr[end-1] # second-highest
        xh, yh = S[end],   y_arr[end]   # highest

        # compute mean
        xm = mean(S[1:end-1])
        xr = xm + α*(xm - xh) # reflection point
        yr = f(xr)

        # apply simplex update
        if yr < yl
            xe = xm + β*(xr - xm) # expansion point
            ye = f(xe)
            S[end], y_arr[end] = ye < yr ? (xe, ye) : (xr, yr)
        elseif yr >= ys
            if yr <= yh
                xh, yh, S[end], y_arr[end] = xr, yr, xr, yr
            end
            xc = xm + γ*(xh - xm) # contraction point
            yc = f(xc)
            if yc > yh
                for i in 2:length(y_arr) # shrinkage
                    S[i] = (S[i] + xl)/2
                    y_arr[i] = f(S[i])
                end
            else
                S[end], y_arr[end] = xc, yc
            end
        else
            S[end], y_arr[end] = xr, yr
        end

        Δ = std(y_arr, corrected=false)
        push!(S_history, S)

    end

    return S[argmin(y_arr)], S_history

end

# Nelder-Mead simplex method example

f(x)  = (x[1]+2x[2]-7)^2 + (2x[1]+x[2]-5)^2
S     = [[10., 10.], [0., -10.], [-10., 0.]]
x, Sh = nelder_mead_simplex_method(f, S, 1e-3)

"""
contourf(-10:0.2:10, -10:0.2:15, (x,y) -> f([x,y]), aspect_ratio=:equal, box=:on,
            c=:vibrant_grad_r)

for i in 1:length(Sh)

    simplex = Shape( [Tuple(Sh[i][j][k] for k in 1:length(Sh[i][j])) for j in [1, 2, 3, 1] ] )
    plot!(simplex, c=false, lc=:black, linealpha = i/length(Sh), label=:none)

end

xlims!(-10,10)
ylims!(-10,15)
"""

## divided rectangles (DIRECT) method
# adapted from example notebook since this algorithm is complicated

struct Interval
    c
    y
    depths
end

function rev_unit_hypercube_parameterization(x, a, b)
    return x.*(b-a) + a
end

function reparameterize_to_unit_hypercube(f, a, b)
    Δ = b-a
    return x->f(x.*Δ + a)
end

function min_depth(I)
    return minimum(I.depths)
end

function add_interval!(intervals, I)

	d = min_depth(I)

    if !haskey(intervals, d)
        intervals[d] = PriorityQueue{Interval, Float64}()
    end

    enqueue!(intervals[d], I, I.y)

end

function get_opt_intervals(intervals, ϵ, y_best)

    max_depth = maximum(keys(intervals))
    stack = [DataStructures.peek(intervals[max_depth])[1]]
    d = max_depth-1

    while d ≥ 0

        if haskey(intervals, d) && !isempty(intervals[d])

            I = DataStructures.peek(intervals[d])[1]
            x, y = 0.5*3.0^(-min_depth(I)), I.y

            while !isempty(stack)

            	I1 = stack[end]
            	x1, y1 = 0.5*3.0^(-min_depth(I1)), I1.y
            	L1 = (y - y1)/(x - x1)

            	if y1 - L1*x1 > y_best - ϵ || y < y1
                    pop!(stack)
            	elseif length(stack) > 1

            		I2 = stack[end-1]
            		x2, y2 = 0.5*3.0^(-min_depth(I2)), I2.y
            		L2 = (y1 - y2)/(x1 - x2)

            		if L2 > L1
            			pop!(stack)
                    else
                        break
            		end

                else
                    break
            	end
            end

            push!(stack, I) # add new point

        end

        d -= 1

    end

    return stack

end

const Intervals = Dict{Int, PriorityQueue{Interval, Float64}}

function divide(f, I)

    c, d, n = I.c, min_depth(I), length(I.c)
    dirs = findall(I.depths .== d)
    cs = [(c + 3.0^(-d-1)*basis(i,n),
           c - 3.0^(-d-1)*basis(i,n)) for i in dirs]
    vs = [(f(C[1]), f(C[2])) for C in cs]
    minvals = [min(V[1], V[2]) for V in vs]

    retval = Interval[]
    depths = copy(I.depths)

    for j in sortperm(minvals)

        depths[dirs[j]] += 1
        C, V = cs[j], vs[j]
        push!(retval, Interval(C[1], V[1], copy(depths)))
        push!(retval, Interval(C[2], V[2], copy(depths)))

    end

    push!(retval, Interval(c, I.y, copy(depths)))

    return retval

end

function direct_method(f, a, b, ϵ, K)

    g = reparameterize_to_unit_hypercube(f, a, b)
    intervals = Intervals()
    n = length(a)
    c = fill(0.5, n)
    I = Interval(c, g(c), fill(0, n))
    add_interval!(intervals, I)
    c_best, y_best = copy(I.c), I.y

    for k in 1 : K

        S = get_opt_intervals(intervals, ϵ, y_best)
        to_add = Interval[]

        for I in S

            append!(to_add, divide(g, I))
            dequeue!(intervals[min_depth(I)])

        end

        for I in to_add

            add_interval!(intervals, I)

            if I.y < y_best
                c_best, y_best = copy(I.c), I.y
            end

        end

    end

    return rev_unit_hypercube_parameterization(c_best, a, b)

end

## DIRECT method example

f(x)  = (x[1]+2x[2]-7)^2 + (2x[1]+x[2]-5)^2
x = direct_method(f, [10, 10], [-10, -10], 1e-2, 25)
