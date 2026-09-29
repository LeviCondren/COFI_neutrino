using Plots

function plot_ir_solution(pE, phi_interp; z_values=z_grid,
					 reltol=1e-8, abstol=1e-10, kwargs...)
    sol = solve_ir_solution(pE, phi_interp; reltol=reltol, abstol=abstol)
    values = sol.(z_values)
    fL = [value[1] for value in values]
    fR = [value[2] for value in values]

    plot(z_values, fL;
	    label="fL", xlabel="z", ylabel="solution", kwargs...)
    plot!(z_values, fR; label="fR")
end
