module backgroundSolve

export dirac_rhs!, solve_uv_solution, solve_ir_solution, plot_ir_solution, propagator_trace, propagator_trace_direct, renormalized_condensate,
       solve_uv_angle, solve_ir_angle,
       kleinGordon_rhs!, solve_uv_scalar_solution, solve_ir_scalar_solution, get_scalar_components,
       tabulate_condensate, solve_self_consistent_background

const k = 1.0
const z0 = 1.0
const z1 = 100.0

const c_psi = 1.0
const m_psi = c_psi * k
const m_phi = m_psi  # scalar mass set to the same scale as the fermion
const p_max = 10.0 * k
y5 = 0.1  # Not const so you can vary it

# Grid
const N_z = 500
const z_grid = range(z0, z1, length=N_z)

include(joinpath(@__DIR__, "homogeneousEquSolve.jl"))
include(joinpath(@__DIR__, "condensateSolve.jl"))
include(joinpath(@__DIR__, "scalarEOMsolve.jl"))
include(joinpath(@__DIR__, "plot.jl"))
end