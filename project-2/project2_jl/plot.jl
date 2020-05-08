using Plots; pyplot()
using Colors

include("project2.jl")
include("helpers.jl")
include("simple.jl")

simple1_plot(x, y) = simple1([x, y])
simple2_plot(x, y) = log10(max(simple2([x, y]), 0.01))

simple1_constr1(x, y) = simple1_constraints([x, y])[1]
simple1_constr2(x, y) = simple1_constraints([x, y])[2]

simple2_constr1(x, y) = simple2_constraints([x, y])[1]
simple2_constr2(x, y) = simple2_constraints([x, y])[2]

col = colorant"#424242";

xplot = -3:0.01:3
yplot = -3:0.01:3

## optimization plots

function optimization_plot(f_plot, f_constr1, f_constr2, xplot, yplot, f, g, c, x0, n, prob)

    contourf(xplot, yplot, f_plot, aspect_ratio=:equal, size=(600,400), grid=false,
                xlims=(-3, 3), ylims=(-3, 3), clims=(-2,4), #levels=30,
                box=true, color=cgrad(:vibrant_grad_r, scale=:linear))

    contour!(xplot, yplot, f_constr1, levels=[0, 0.001], w=1.5, c=:black)
    contour!(xplot, yplot, f_constr2, levels=[0, 0.001], w=1.5, c=:black)

    for i in 1:3

        for key in keys(COUNTERS)
            COUNTERS[key] = 0
        end

        history = [:]
        history = optimize(f, g, c, x0(), n, prob)

        plot!(getindex.(history[:],1), getindex.(history[:],2),
                c=col, m=:o, ms=2, mc=:white, msc=col, w=1, label=:none)
        scatter!([history[1][1]], [history[1][2]],
                c=col, m=:o, ms=3, mc=:black, msc=col, w=1, label=:none)
    end

    plot!()

end

# simple 1
optimization_plot(simple1_plot, simple1_constr1, simple1_constr2, xplot, yplot, simple1,
                simple1_gradient, simple1_constraints, simple1_init, 5000, "simple1")
savefig("plots/simple1_alg1_new.svg")

# simple 2
optimization_plot(simple2_plot, simple2_constr1, simple2_constr2, xplot, yplot, simple2,
                simple2_gradient, simple2_constraints, simple2_init, 10000, "simple2")
savefig("plots/simple2_alg1_new.svg")


# need to make modified optimize function to run a second case
# then need to do constraint violation plot

# function optimization_plot(f_plot, descent_method, f, g, x0, n, x, y, lims)
#
#     contourf(x, y, f_plot, aspect_ratio=:equal, size=(600,400), grid=false,
#                 xlims=(-lims,lims), ylims=(-lims,lims), clims=(-1,3),
#                 levels=20, box=true, color=cgrad(:vibrant_grad_r, scale=:linear))
#
#     for i in 1:3
#         history = optimize_w_history(descent_method, f, g, x0(), n)
#         plot!(getindex.(history[:],1), getindex.(history[:],2),
#                 c=col, m=:o, ms=2, mc=:white, msc=col, w=1, label=:none)
#         scatter!([history[1][1]], [history[1][2]],
#                 c=col, m=:o, ms=3, mc=:black, msc=col, w=1, label=:none)
#     end
#
#     plot!()
#
# end
#
#
# # rosenbrock
# optimization_plot(rosenbrock_plot, descent_method, rosenbrock_unc,
#                     rosenbrock_gradient_unc, rosenbrock_init_unc, 100, x, y, 2)
# savefig("rosenbrock_opt.svg")
#
#
# # himmelblau
# optimization_plot(himmelblau_plot, descent_method, himmelblau_unc,
#                     himmelblau_gradient_unc, himmelblau_init_unc, 50, x, y, 5)
# savefig("himmelblau_opt.svg")
#
#
# ## convergence plots
#
# function convergence_plot(y)
#
#     plot(1:2:length(y), y[1:2:end], size=(400,400), box=true,
#             xlabel="Iteration", ylabel="Absolute Error",
#             c=col, m=:o, ms=2, mc=:white, msc=col, w=1,
#             label=:none, yscale=:log10)
#
# end
#
#
# # rosenbrock
# rosenbrock_history = optimize_w_history(descent_method, rosenbrock_unc,
#                             rosenbrock_gradient_unc, rosenbrock_init_unc(), 200)
# y = rosenbrock.(rosenbrock_history)
#
# convergence_plot(y)
# savefig("rosenbrock_conv_3.svg")
#
#
# # himmelblau
# himmelblau_history = optimize_w_history(descent_method, himmelblau_unc,
#                             himmelblau_gradient_unc, himmelblau_init_unc(), 200)
# y = himmelblau.(himmelblau_history)
#
# convergence_plot(y)
# savefig("himmelblau_conv_3.svg")
#
#
# # powell
# powell_history = optimize_w_history(descent_method, powell_unc,
#                             powell_gradient_unc, powell_init_unc(), 200)
# y = powell.(powell_history)
#
# convergence_plot(y)
# savefig("powell_conv_3.svg")
