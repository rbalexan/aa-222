f(x)  =   x[1]^2 + x[2]^2 + 0.1x[2]^4
∇f(x) = [2x[1],   2x[2]+0.4x[2]^3]
x     = [5, 4]

hgdwnm_method = HypergradientDescentWithNesterovMomentum(0.001, 1e-5, 0.5, 0, 0.001, 0)
init!(hgdwnm_method, f, ∇f, x)
x_hist = []

for i in 1:10
    global x = iterate!(hgdwnm_method, f, ∇f, x, 1)
    push!(x_hist, x)
end
