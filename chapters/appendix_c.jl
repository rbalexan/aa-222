using Roots

## Gaussian quadrature

struct Quadrule
    ws
    xs
end

function legendre(i)

    # generate anonymous Legendre polynomial of order i

    f(x) = 1

    if i == 0.
        return x -> 1
    elseif i == 1.
        return x -> x
    elseif i == 2.
        return x -> 3/2*x^2 - 1/2
    elseif i == 3.
        return x -> 5/2*x^3 - 3/2*x
    end

end

function quadrature_rule_legendre(m)

    # generate the Legendre polynomials and solve for the quadrature points
    bs = [legendre(i) for i in 0:m]
    xs = find_zeros(bs[end], -1, 1)

    # evaluate the orthogonal polynomials at the quadrature points
    # and solve for the weights
    A = [bs[k](xs[i]) for k in 1:m, i in 1:m]
    b = zeros(m)
    b[1] = 2
    ws = A\b

    return Quadrule(ws, xs)

end

function quadint(f, quadrule)

    return sum(w*f(x) for (w,x) in zip(quadrule.ws, quadrule.xs))

end

function quadint(f, quadrule, a, b)

    # transform the integral to [-1, 1]
    α = (b-a)/2
    β = (a+b)/2
    g = x -> α*f(α*x+β)

    return quadint(g, quadrule)

end

# example Gaussian quadrature

three_pt_quadrule = quadrature_rule_legendre(3)

f(x) = x^5 - 2x^4 + 3x^3 + 5x^2 - x + 4
quadint(f, three_pt_quadrule)
quadint(f, three_pt_quadrule, -3, 5)
