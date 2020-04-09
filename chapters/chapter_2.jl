using SymEngine
using LinearAlgebra # for ⋅
      LinearAlgebra.conj(b::Basic) = b # only valid if real
using Test
using ForwardDiff

## gradient and hessian operations
create_vars(variable::String, dims::Int) = [symbols("$(variable)$i") for i in 1:dims]

∇( f::Basic, vars::Vector{Basic}) = diff.(f, collect(Iterators.product(vars)))
∇²(f::Basic, vars::Vector{Basic}) = diff.(f, collect(Iterators.product(vars, vars)))

# function f
f_vars = create_vars("x", 3)
f = 2x1 -x3^2 + sin(x1) + x1*cos(x2)

∇(f, f_vars)
∇²(f, f_vars)

# function g
g_vars = create_vars("x", 2)
g = x1*sin(x2) + 1

∂g∂x = ∇(g, vars)
∂g∂x = subs.(∂g∂x, x1=>2, x2=>0)

# function h
h_vars = create_vars("x", 2)
h = x1*x2

∂h∂x = ∇(h, vars)
∂h∂x = subs.(∂h∂x, x1=>1, x2=>0)

s = [-1, -1]
∂h∂s = ∂h∂x⋅s

## finite difference techniques
diff_forward( f, x; h=sqrt(eps(Float64))) = (f(x+h)   - f(x))     / h
diff_central( f, x; h=cbrt(eps(Float64))) = (f(x+h/2) - f(x-h/2)) / h
diff_backward(f, x; h=sqrt(eps(Float64))) = (f(x)     - f(x-h))   / h
diff_complex( f, x; h=1e-20)              = imag(f(x+h*im))       / h

function complex_step( f, x, h=1e-20)
    v = f(x+h*im)
    return (real(v), imag(v)/h)
end

# function i
i    = x -> sin(x^2)
didx = x -> 2*x*cos(x^2)

(cs_value, cs_deriv) = complex_step(i, π/2)

@test i(   π/2) ≈ value
@test didx(π/2) ≈ deriv

## automatic differentiation using forward accumulation
struct Dual
    v
    ∂
end

# overload Base functions
Base.:+( a::Dual, b::Dual) = Dual(a.v + b.v, a.∂ + b.∂)
Base.:*( a::Dual, b::Dual) = Dual(a.v * b.v, a.v*b.∂ + b.v*a.∂)
Base.log(a::Dual)          = Dual(log(a.v), a.∂/a.v)
function Base.max(a::Dual, b::Dual)
    v = max(a.v, b.v)
    ∂ = a.v > b.v ? a.∂ : a.v < b.v ? b.∂ : NaN # NaN if equal
    return Dual(v, ∂)
end
function Base.max(a::Dual, b::Int)
    v = max(a.v, b)
    ∂ = a.v > b ? a.∂ : a.v < b ? 0 : NaN # NaN if equal
    return Dual(v, ∂)
end

a = Dual(3, 1)
b = Dual(2, 0)

log(a*b + max(a,2))

# using ForwardDiff Dual type
a = ForwardDiff.Dual(3,1)
b = ForwardDiff.Dual(2,0)

log(a*b + max(a,2))

## automatic differentiation using reverse accumulation
import Zygote: gradient
f(a, b) = log(a*b + max(a,2))
gradient(f, 3.0, 2.0)
