using Random

include("chapter_5.jl") # for basic descent methods

## noisy descent

mutable struct NoisyDescent <: DescentMethod
    submethod
    σ # noise sequence function
    k # counter
end

function init!(M::NoisyDescent, f, ∇f, x)

    # initalize submethod and set counter to 1
    init!(M.submethod, f, ∇f, x)
    M.k = 1

    return M

end

function step!(M::NoisyDescent, f, ∇f, x)

    x = step!(M.submethod, f, ∇f, x)
    σ = M.σ(M.k)
    x += σ.*randn(length(x))
    M.k += 1

    return x

end

# noisy descent example

f(x)  =   x[1]^2 + x[2]^2 + 0.1x[2]^4
∇f(x) = [2x[1],   2x[2]+0.4x[2]^3]
x     = [5, 4]
σ(k)  = 1/k

gd_method = GradientDescent(0.1)
init!(ngd_method, f, ∇f, x)
x = iterate!(ngd_method, f, ∇f, x, 6)

Random.seed!(1)
ngd_method = NoisyDescent(gd_method, σ, 0)
init!(ngd_method, f, ∇f, x)
x = iterate!(ngd_method, f, ∇f, x, 6)

## mesh adaptive direct search
