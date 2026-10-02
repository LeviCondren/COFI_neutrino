using DifferentialEquations
using Interpolations

# ==================================================================
# Klein-Gordon equation for the bulk scalar background phi(z) (Euclidean,
# zero boundary momentum), sourced by the fermion condensate <psibar psi>(z):
#
#   z^2 phi'' - 3 z phi' - m_phi^2 phi = y5 * <psibar psi>(z)
#
# Homogeneous solutions z^Δ± with Δ± = 2 ± sqrt(4 + m_phi^2).
# Solved as a first-order system with state y = [phi, dphi/dz]:
#
#   dy[1] = dphi
#   dy[2] = (3 z dphi + m_phi^2 phi + y5 * source(z)) / z^2
#
# m_phi and y5 are global constants (see backgroundSolve.jl). The pE
# argument of the solvers below is unused (kept so call sites don't change).
# `source_interp` is a fast
# interpolant standing in for the condensate <psibar psi>(z) - see
# tabulate_condensate below for why it's precomputed rather than called
# directly here.
# ==================================================================

function kleinGordon_rhs!(dy, y, params, z)
    source_interp = params[1]
    phi, dphi = y[1], y[2]

    # z^2 ddphi - 3 z dphi - m_phi^2 phi - y5 * source_interp(z) = 0,
    # solved for ddphi = phi'':
    ddphi = (3 * z * dphi + m_phi^2 * phi + y5 * source_interp(z)) / z^2

    dy[1] = dphi    # d(phi)/dz
    dy[2] = ddphi   # d(dphi)/dz = phi''
end


# ==================================================================
# Solvers
# ==================================================================

"""
Solve with UV boundary conditions:
    phi(z0)  = 1
    dphi(z0) = 0  (normalization)

`source_interp` is the (interpolated) condensate <psibar psi>(z) sourcing
the equation - see tabulate_condensate.
"""
function solve_uv_scalar_solution(pE, source_interp; reltol=1e-8, abstol=1e-10)
    y0 = [1.0, 0.0]
    zspan = (z0, z1)
    params = (source_interp,)

    prob = ODEProblem(kleinGordon_rhs!, y0, zspan, params)
    sol = solve(prob, Vern9(),
                reltol=reltol, abstol=abstol,
                dense=true)

    return sol
end

"""
Solve with IR boundary conditions:
    phi(z1)  = 0
    dphi(z1) = 1  (normalization)

Integrates BACKWARD from z1 to z0.
"""
function solve_ir_scalar_solution(pE, source_interp; reltol=1e-8, abstol=1e-10)
    y0 = [0.0, 1.0]
    zspan = (z1, z0)   # Backward integration
    params = (source_interp,)

    prob = ODEProblem(kleinGordon_rhs!, y0, zspan, params)
    sol = solve(prob, Vern9(),
                reltol=reltol, abstol=abstol,
                dense=true)

    return sol
end


# ==================================================================
# Convenience: extract phi, dphi from a solution
# ==================================================================

"""
Extract phi(z) and dphi(z) from a scalar ODE solution.

Usage:
    sol = solve_uv_scalar_solution(1.0, source_interp)
    phi, dphi = get_scalar_components(sol)
    phi(10.0)  # evaluates phi at z=10
"""
function get_scalar_components(sol)
    phi  = z -> sol(z)[1]
    dphi = z -> sol(z)[2]
    return phi, dphi
end


# ==================================================================
# Condensate tabulation
# ==================================================================

"""
renormalized_condensate(z, phi_interp, p_max) costs ~1-2 minutes PER CALL
(it solves the full Dirac UV+IR problem inside a quadgk integral). An
adaptive ODE solver calls its right-hand side many times per step, so
kleinGordon_rhs! must never call renormalized_condensate directly.

Instead, evaluate it on a coarse z grid once and build a fast interpolant
to hand to solve_uv_scalar_solution / solve_ir_scalar_solution as
`source_interp`.

n_points=20 -> ~20 x (1-2 min) per call to this function. Keep n_points
small; raise it only once the rest of the pipeline is validated.

NOTE: each z point is independent in principle, but DO NOT parallelize this
loop with Threads.@threads - the DifferentialEquations.jl/QuadGK call chain
inside renormalized_condensate is not safe to call concurrently from
multiple threads sharing one process (confirmed: it throws
`ConcurrencyViolationError("Vector can not be resized concurrently")`
under load). For real parallelism, use process-based parallelism
(Distributed.jl) instead - see run_selfconsistent_checkpointed.jl.
"""
function tabulate_condensate(phi_interp, p_max; n_points=20, rtol=1e-3, verbose=true)
    zs = range(z0, z1, length=n_points)
    vals = Vector{Float64}(undef, n_points)
    for (i, z) in enumerate(zs)
        vals[i] = renormalized_condensate(z, phi_interp, p_max; rtol=rtol)
        verbose && println("  tabulate_condensate: z = ", z, "  <psibar psi> = ", vals[i])
    end
    return linear_interpolation(zs, vals; extrapolation_bc=Line())
end


# ==================================================================
# Self-consistent background (fixed-point iteration)
# ==================================================================

"""
Solve the coupled scalar-fermion background self-consistently:
  - phi(z) sources the fermion mass term (y5 * phi_interp(z) in dirac_rhs!)
  - the fermion condensate <psibar psi>(z) sources phi's own equation of
    motion (y5 * source(z) in kleinGordon_rhs!)

Initial guess (pass 0): the UV solution of the plain homogeneous equation,
source turned OFF (source_interp = z -> 0). This is a real, principled
solution of the equation - not an arbitrary function - just without any
fermion backreaction yet.

From pass 1 onward, the source is turned back ON: tabulate the condensate
from the current phi_interp -> solve the scalar EOM sourced by it -> take
the UV solution as the next phi_interp -> repeat until the profile stops
changing (max pointwise change on a small sample of z below `tol`), or
`max_iter` is reached.

CAVEAT: the UV/IR pair here is built the same way as the fermion
propagator (independent [1,0] / [0,1] boundary data at each end), which is
the right structure for a Green's function. A genuine classical background
profile is normally a single solution fixed by *one* condition at each end
(e.g. a chosen UV source and IR regularity), not an arbitrary combination
of two independently-normalized pieces. This iteration uses the UV
solution alone as the candidate profile as a starting point - flag this if
it's not the right boundary-value structure for what you want physically.
"""
function solve_self_consistent_background(pE, p_max; max_iter=10, tol=1e-4,
                                           n_points=20, rtol=1e-3, verbose=true)
    sample_zs = range(z0, z1, length=5)

    verbose && println("=== pass 0: homogeneous solve, source off ===")
    phi_interp, _ = get_scalar_components(solve_uv_scalar_solution(pE, z -> 0.0))

    for iter in 1:max_iter
        verbose && println("=== self-consistency iteration ", iter, " (source on) ===")
        source_interp = tabulate_condensate(phi_interp, p_max; n_points=n_points, rtol=rtol, verbose=verbose)

        uv_sol = solve_uv_scalar_solution(pE, source_interp)
        new_phi_interp, _ = get_scalar_components(uv_sol)

        max_diff = maximum(abs(new_phi_interp(z) - phi_interp(z)) for z in sample_zs)
        verbose && println("iteration ", iter, ": max change on sample = ", max_diff)

        phi_interp = new_phi_interp
        if max_diff < tol
            verbose && println("Converged after ", iter, " iterations.")
            break
        end
    end

    return phi_interp
end
