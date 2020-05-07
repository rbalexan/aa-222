## gradient descent with Nesterov momentum

mutable struct NesterovMomentum <: UnconstrainedOptimizationMethod
    α::Float64 # learning rate
    β::Float64 # momentum decay
    v::Vector # momentum
end


function init!(M::NesterovMomentum, f::Function, g::Function, x::Vector)

    # make momentum vector
    M.v = zeros(length(x))
    return M

end


function step!(M::NesterovMomentum, f::Function, g::Function, x::Vector)

    # access parameters
    α, β, v = M.α, M.β, M.v

    # update Nesterov momentum and return new x value
    v[:] = β*v - α*g(x + β*v)
    x′ = x + v

    return x′

end
