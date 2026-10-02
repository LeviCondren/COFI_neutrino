using DifferentialEquations
using Interpolations

# ==================================================================
# ODE right-hand side
# ==================================================================

"""
The coupled Dirac ODE system in explicit form:
    dfL/dz = -M(z) fL + pE fR
    dfR/dz =  M(z) fR - pE fL

y = [fL, fR]
"""
function dirac_rhs!(dy, y, params, z)
    pE, phi_interp = params
    
    M = (m_psi + y5 * phi_interp(z)) / (k * z)
    
    fL, fR = y[1], y[2]
    dy[1] = -M * fL + pE * fR
    dy[2] =  M * fR + pE * fL
end


# ==================================================================
# Solvers
# ==================================================================

"""
Solve with UV boundary conditions:
    fR(z0) = 0
    fL(z0) = 1  (normalization)

Returns an ODE solution object; call sol(z) to evaluate at any z.
"""
function solve_uv_solution(pE, phi_interp; reltol=1e-8, abstol=1e-10)
    # Initial conditions at z0: [fL, fR] = [1, 0]
    y0 = [1.0, 0.0]
    zspan = (z0, z1)
    params = (pE, phi_interp)
    
    prob = ODEProblem(dirac_rhs!, y0, zspan, params)
    sol = solve(prob, Vern9(),  # 9th-order Verner method; very accurate
                reltol=reltol, abstol=abstol,
                dense=true)      # Continuous interpolation
    
    return sol
end

"""
Solve with IR boundary conditions:
    fL(z1) = 0
    fR(z1) = 1  (normalization)

Integrates BACKWARD from z1 to z0.
"""
function solve_ir_solution(pE, phi_interp; reltol=1e-8, abstol=1e-10)
    y0 = [0.0, 1.0]
    zspan = (z1, z0)   # Backward integration
    params = (pE, phi_interp)
    
    prob = ODEProblem(dirac_rhs!, y0, zspan, params)
    sol = solve(prob, Vern9(),
                reltol=reltol, abstol=abstol,
                dense=true)
    
    return sol
end


# ==================================================================
# Convenience: extract fL, fR from a solution
# ==================================================================

"""
Extract fL(z) and fR(z) from an ODE solution.

Usage:
    sol = solve_uv_solution(1.0, phi_interp)
    fL, fR = get_components(sol)
    fL(10.0)  # evaluates fL at z=10
"""
function get_components(sol)
    #Multiply by (k*z)^2 to get the correctly scaled components
    fL = z -> (k*z)^2 * sol(z)[1]
    fR = z -> (k*z)^2 * sol(z)[2]
    return fL, fR
end

# ==================================================================
# Overflow-free (Prüfer angle) form
# ==================================================================

"""
Write (fL, fR) = r (cos θ, sin θ). The trace in propagator_trace depends only
on the ratio fR/fL of each solution, i.e. on θ, and (Euclidean dirac_rhs!) θ obeys

    dθ/dz = M(z) sin(2θ) + pE cos(2θ)

which stays O(1) however large M gets - unlike fL, fR, which grow like
exp(∫2M) or exp(pE Δz) and overflow Float64 (|phi| ~ 1e6, or pE >~ 8 at phi = 0).
UV: (fL, fR)(z0) = (1, 0) -> θ(z0) = 0.  IR: (fL, fR)(z1) = (0, 1) -> θ(z1) = π/2.

Large |M| makes this stiff, hence the auto-switching stiff solver.
"""
function dirac_angle_rhs(θ, params, z)
    pE, phi_interp = params
    M = (m_psi + y5 * phi_interp(z)) / (k * z)
    return M * sin(2θ) + pE * cos(2θ)
end

function solve_uv_angle(pE, phi_interp; reltol=1e-10, abstol=1e-12)
    prob = ODEProblem(dirac_angle_rhs, 0.0, (z0, z1), (pE, phi_interp))
    return solve(prob, AutoVern9(Rodas5P()); reltol=reltol, abstol=abstol, dense=true)
end

function solve_ir_angle(pE, phi_interp; reltol=1e-10, abstol=1e-12)
    prob = ODEProblem(dirac_angle_rhs, π/2, (z1, z0), (pE, phi_interp))
    return solve(prob, AutoVern9(Rodas5P()); reltol=reltol, abstol=abstol, dense=true)
end
