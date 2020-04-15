## gradient descent with Nesterov momentum

mutable struct GradientDescentWithNesterovMomentum <: DescentMethod
    α # learning rate
    β # momentum decay
    v # momentum
end

function init!(M::GradientDescentWithNesterovMomentum, f, ∇f, x)

    # make momentum vector
    M.v = zeros(length(x))
    return M

end

function step!(M::GradientDescentWithNesterovMomentum, f, ∇f, x)

    # access parameters
    α, β, v = M.α, M.β, M.v

    # update Nesterov momentum and return new x value
    v[:] = β*v - α*∇f(x + β*v)
    x′ = x + v

    return x′

end
