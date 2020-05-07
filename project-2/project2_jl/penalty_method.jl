## penalty method

mutable struct PenaltyMethod <: ConstrainedOptimizationMethod
    ρ1::Float64
    ρ2::Float64
    γ::Float64
    penalty::Function
    penalty_gradient::Function
    unconstrained_opt_method::UnconstrainedOptimizationMethod
    return_first_feasible::Bool
end


function step!(M::PenaltyMethod, f::Function, g::Function, c::Function, x::Vector)

    # step using modified penalty function and finite difference gradient

    fp(x) = f(x) + M.penalty(M, c, x)
    gp(x) = g(x) + M.penalty_gradient(M, c, x)

    x′ = step!(M.unconstrained_opt_method, x -> fp(x), x -> gp(x), x)

    M.ρ1 *= M.γ
    M.ρ2 *= M.γ

    # return first feasible
    if M.return_first_feasible && M.penalty(M, c, x′) == 0
        return x′, true
    else
        return x′, false
    end

end


## penalties

function count_penalty(M::PenaltyMethod, c::Function, x::Vector)

    constraint_values = c(x)
    penalty           = size(constraint_values[constraint_values .> 0])[1]

    return M.ρ1*penalty

end


function quadratic_penalty(M::PenaltyMethod, c::Function, x::Vector)

    constraint_values = c(x)
    penalty           = sum(max.(constraint_values, 0).^2)

    return M.ρ1*penalty

end


function mixed_penalty(M::PenaltyMethod, c::Function, x::Vector)

    constraint_values = c(x)
    count_penalty     = size(constraint_values[constraint_values .> 0])[1]
    quadratic_penalty = sum(max.(constraint_values, 0).^2)

    return M.ρ1*count_penalty + M.ρ2*quadratic_penalty

end


## penalty gradients

function forward_difference_penalty_gradient(M::PenaltyMethod, c::Function, x::Vector)

    dim = size(x)[1]
    g   = zeros(dim)

    basis(i) = [i == j for j in 1:dim]

    f_penalty_x0 = M.penalty(M, c, x)

    for i in 1:dim

        f_penalty_x1 = M.penalty(M, c, x+(eps()*basis(i)))
        g[i] = (f_penalty_x1 - f_penalty_x0) / eps()

    end

    return g

end
