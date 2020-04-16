## adaptive delta (adadelta)

mutable struct Adadelta <: DescentMethod
    γs # gradient decay factor
    γx # update decay factor
    ϵ # small value
    s # sum of squared gradients
    u # sum of squared updates
end

function init!(M::Adadelta, f, ∇f, x)

    # make squared gradients and squared updates vectors
    M.s = zeros(length(x))
    M.u = zeros(length(x))
    return M

end

function step!(M::Adadelta, f, ∇f, x)

    # access parameters and compute gradient
    γs, γx, ϵ, s, u, g = M.γs, M.γx, M.ϵ, M.s, M.u, ∇f(x)

    # update sum of squared gradients and sum of squared updates
    # and return new x value
    s[:] = γs*s + (1-γs)*(g.*g)
    Δx = - (sqrt.(u) .+ ϵ) ./ (sqrt.(s) .+ ϵ) .* g
    u[:] = γx*u + (1-γx)*(Δx.*Δx)
    x′ = x + Δx

    return x′

end
