using Plots; pyplot()
using Colors

include("project1.jl")
include("helpers.jl")
include("simple.jl")

rosenbrock_plot(x, y) = log10(max(rosenbrock([x,y]),0.01))
himmelblau_plot(x, y) = log10(max(himmelblau([x,y]),0.01))

descent_method = Adam(2e-1, 0.7, 0.99, 1e-6, 0, 0, 0)
#descent_method = GradientDescentWithNesterovMomentum(1e-3, 0.95, 0)

col = colorant"#424242";

x = -5:0.01:5
y = -5:0.01:5

## optimization plots

function optimization_plot(f_plot, descent_method, f, g, x0, n, x, y, lims)

    contourf(x, y, f_plot, aspect_ratio=:equal, size=(600,400), grid=false,
                xlims=(-lims,lims), ylims=(-lims,lims), clims=(-1,3),
                levels=20, box=true, color=cgrad(:vibrant_grad_r, scale=:linear))

    for i in 1:3
        history = optimize_w_history(descent_method, f, g, x0(), n)
        plot!(getindex.(history[:],1), getindex.(history[:],2),
                c=col, m=:o, ms=2, mc=:white, msc=col, w=1, label=:none)
        scatter!([history[1][1]], [history[1][2]],
                c=col, m=:o, ms=3, mc=:black, msc=col, w=1, label=:none)
    end

    plot!()

end


# rosenbrock
optimization_plot(rosenbrock_plot, descent_method, rosenbrock_unc,
                    rosenbrock_gradient_unc, rosenbrock_init_unc, 100, x, y, 2)
savefig("rosenbrock_opt.svg")


# himmelblau
optimization_plot(himmelblau_plot, descent_method, himmelblau_unc,
                    himmelblau_gradient_unc, himmelblau_init_unc, 50, x, y, 5)
savefig("himmelblau_opt.svg")


## convergence plots

function convergence_plot(y)

    plot(1:2:length(y), y[1:2:end], size=(400,400), box=true,
            xlabel="Iteration", ylabel="Absolute Error",
            c=col, m=:o, ms=2, mc=:white, msc=col, w=1,
            label=:none, yscale=:log10)

end


# rosenbrock
rosenbrock_history = optimize_w_history(descent_method, rosenbrock_unc,
                            rosenbrock_gradient_unc, rosenbrock_init_unc(), 200)
y = rosenbrock.(rosenbrock_history)

convergence_plot(y)
savefig("rosenbrock_conv_3.svg")


# himmelblau
himmelblau_history = optimize_w_history(descent_method, himmelblau_unc,
                            himmelblau_gradient_unc, himmelblau_init_unc(), 200)
y = himmelblau.(himmelblau_history)

convergence_plot(y)
savefig("himmelblau_conv_3.svg")


# powell
powell_history = optimize_w_history(descent_method, powell_unc,
                            powell_gradient_unc, powell_init_unc(), 200)
y = powell.(powell_history)

convergence_plot(y)
savefig("powell_conv_3.svg")
