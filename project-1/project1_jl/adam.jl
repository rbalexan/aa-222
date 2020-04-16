## adaptive moment estimation (adam)

mutable struct Adam <: DescentMethod
    α # learning rate
    γv # 1st moment decay factor
    γs # 2nd moment decay factor
    ϵ # small value
    k # step counter
    v # 1st moment estimate
    s # 2nd moment estimate
end

function init!(M::Adam, f, ∇f, x)

    # ensure k=0 and make 1st and 2nd moment vectors
    M.k = 0
    M.v = zeros(length(x))
    M.s = zeros(length(x))
    return M

end

function step!(M::Adam, f, ∇f, x)

    # access parameters and compute gradient
    α, γv, γs, ϵ, k, v, s, g = M.α, M.γv, M.γs, M.ϵ, M.k, M.v, M.s, ∇f(x)

    # update first and second moment estimates
    v[:] = γv*v + (1-γv)*(g)
    s[:] = γs*s + (1-γs)*(g.*g)
    M.k = k += 1

    # perform bias correction
    v_hat = v ./ (1 - γv^k)
    s_hat = s ./ (1 - γs^k)

    # update and return new x value
    x′ = x - α*v_hat ./ (sqrt.(s_hat) .+ ϵ)

    return x′

end
