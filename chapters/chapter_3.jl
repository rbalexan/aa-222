using Base.MathConstants: φ
using Plots; pyplot()

## determining an initial bracket

"""
    bracket_minimum(f, x)
Determine a suitable initial bracket for a unimodal function `f` given an estimate `x`.
"""
function bracket_minimum(f, x=0; s=1e-3, k=2.0)

    # initialize bracket bounds
    a, ya = x,     f(x)
    b, yb = a + s, f(a + s)

    # reverse search direction if necessary
    if yb > ya
        a,  b  = b,  a
        ya, yb = yb, ya
        s = -s
    end

    # perform bracket expansion
    while true

        # update bracket bound
        c, yc = b + s, f(b + s)

        # return if new function value increases
        if yc > yb
            return a < c ? (a, c) : (c, a)
        end

        # update bracket bounds and step size
        a,  b  = b,  c  # b is a betten bound than a; c is a better bound than b
        ya, yb = yb, yc
        s *= k

    end

end

# bracket minimum example

f = x -> x^2
(l, u) = bracket_minimum(f, 0.2)

## fibonacci search

"""
    fibonacci_search(f, a, b, n)
Perform Fibonacci search on `f` in the interval [`a`,`b`] with `n` evaluations.
"""
function fibonacci_search(f, a, b, n; ϵ=1e-2)

    # define fibonacci ratios
    s = (1-√5) / (1+√5)
    ρ = 1 / (φ*(1 - s^(n+1))/(1 - s^n)) # note: this is inverse of eq. 3.3

    # perform first evaluation (at ρ along interval)
    d  = a + ρ*(b-a)
    yd = f(d)

    # iterate
    for i in 1:n-1

        # compute new bound (last iteration is incremented by ϵ) and perform evaluation
        c  = i ≠ n-1 ? b - ρ*(b-a) : d - ϵ*(d-a)
        yc = f(c)

        # update bounds
        if yc < yd
            b, d, yd = d, c, yc
        else
            a, b = b, c
        end

        # update inverse fibonacci ratio
        ρ = 1 / (φ*(1 - s^(n+1-i))/(1 - s^(n-i)))

    end

    return a < b ? (a, b) : (b, a)

end

# fibonacci search sample function

f = x -> exp(x-2) - x
(l, u) = fibonacci_search(f, -2, 6, 5)

## golden section search

"""
    golden_section_search(f, a, b, n)
Perform golden section search on `f` in the interval [`a`,`b`] with `n` evaluations.
"""
function golden_section_search(f, a, b, n; ϵ=1e-2)

    # define golden section ratio
    ρ = φ-1

    # perform first evaluation (at ρ along interval)
    d  = a + ρ*(b-a)
    yd = f(d)

    # iterate
    for i in 1:n-1

        # compute new bound and perform evaluation
        c  = b - ρ*(b-a)
        yc = f(c)

        # update bounds
        if yc < yd
            b, d, yd = d, c, yc
        else
            a, b = b, c
        end

    end

    return a < b ? (a, b) : (b, a)

end

# golden_section_search sample function

f = x -> exp(x-2) - x
(l, u) = golden_section_search(f, -2, 6, 5)

## quadratic fit search

"""
    golden_section_search(f, a, b, n)
Perform quadratic fit search on `f` in the interval [`a`,`b`] with `n` evaluations.
"""
function quadratic_fit_search(f, a, b, c, n)

    # evaluate at bracket points
    ya, yb, yc = f(a), f(b), f(c)

    # iterate over remaining allocated evalutions
    for i in 1:n-3

        # compute minimum and evaluate
        x = 1/2*(ya*(b^2-c^2) + yb*(c^2-a^2) + yc*(a^2-b^2)) /
                (ya*(b-c)     + yb*(c-a)     + yc*(a-b))
        yx = f(x)

        # update bracketing set
        if x > b
            if yx > yb
                c, yc = x, yx
            else
                a, ya, b, yb = b, yb, x, yx
            end
        elseif x < b
            if yx > yb
                a, ya = x, yx
            else
                c, yc, b, yb = b, yb, x, yx
            end
        end

    end

    return (a, b, c)

end


# quadratic fit search example

f = x -> exp(x-2) - x
(l, m, u) = quadratic_fit_search(f, -2, 3, 6, 10)

## Shubert-Piyavskii method

struct Pt
    x
    y
end

function _get_sp_intersection(A::Pt, B::Pt, l)
    t = ((A.y - B.y) - l*(A.x - B.x)) / 2l
    return Pt(A.x + t, A.y - t*l)
end

"""
    shubert_piyavskii_search(f, a, b, l, ϵ)
Perform Shubert-Piyavskii search on `f` in the interval [`a`,`b`] given a Lipschitz constant `l` and a tolerance `ϵ`.
"""
function shubert_piyavskii_search(f, a, b, l, ϵ, δ=1e-2)

    # generate midpoint and evaluate function at start-, mid-, and end-points
    m = (a + b)/2
    A, M, B = Pt(a, f(a)), Pt(m, f(m)), Pt(b, f(b))

    # compute sawtooth function with intersection points
    pts = [A, _get_sp_intersection(A, M, l),
           M, _get_sp_intersection(M, B, l), B]

    Δ = Inf
    while Δ > ϵ

        # find minimum y point, evaluate it, and update error
        i = argmin([P.y for P in pts])
        P = Pt(pts[i].x, f(pts[i].x))
        Δ = P.y - pts[i].y

        # compute points around P
        P_prev = _get_sp_intersection(pts[i-1], P, l)
        P_next = _get_sp_intersection(P, pts[i+1], l)

        # remove existing point and insert 3 new points
        deleteat!(pts, i)
        insert!(pts, i, P_next)
        insert!(pts, i, P)
        insert!(pts, i, P_prev)

    end

    # compute uncertainty intervals
    intervals = []

    # search function points for minimum y point and return proper sawtooth index
    i = 2*argmin([P.y for P in pts[1:2:end]]) - 1

    # iterate over sawtooth points
    for j in 2:2:length(pts)

        if pts[j].y < pts[i].y # sawtooth point y is below function minimum point y

            # build uncertainty interval
            dy = pts[i].y - pts[j].y
            x_l = max(a, pts[j].x - dy/l)
            x_h = min(b, pts[j].x + dy/l)

            # add intervals and merge if less than tolerance
            if !isempty(intervals) && intervals[end][2] + δ ≥ x_l
                intervals[end] = (intervals[end][1], x_h)
            else
                push!(intervals, (x_l, x_h))
            end

        end

    end

    return (pts[i], intervals)

end

# Shubert Piyavskii search example

f = x -> sin(x)
f_min, intervals = shubert_piyavskii_search(f, 0.1, 2π, 1, 0.01)


## bisection method

"""
    bracket_sign_change(f′, a, b)
Expand the bracket [`a`,`b`] on `f′` until it brackets a zero.
"""
function bracket_sign_change(f′, a, b; k=2)

    # ensure bracket order is correct
    a, b = a < b ? (a, b) : (b, a)

    # iteratively expand bracket until sign change condition is met
    center, half_width = (a+b)/2, (b-a)/2
    while f′(a)*f′(b) > 0
        half_width *= k
        a = center - half_width
        b = center + half_width
    end

    return (a, b)

end

"""
    bisection(f′, a, b, ϵ)
Perform bisection `f′` in [`a`,`b`] until `ϵ`-close to the zero.
"""
function bisection(f′, a, b, ϵ)

    # ensure bracket order is correct
    a, b = a < b ? (a, b) : (b, a)

    # evaluate the endpoints and check if problem is solved
    ya, yb = f′(a), f′(b)
    a = yb == 0 ? b : a
    b = ya == 0 ? a : b

    while b - a > ϵ

        # compute bisection and evaluate function
        x = (a + b)/2
        y = f′(x)

        # update for solution, or select new bounds
        if y == 0
            a, b = x, x
        elseif sign(y) == sign(ya)
            a = x
        else
            b = x
        end

    end

    return (a, b)

end

# bisection method example
f′ = x -> 1 - x
(l, u) = bracket_sign_change(f′, -1, -1.1)
(l, u) = bisection(f′, l, u, 1e-2)
