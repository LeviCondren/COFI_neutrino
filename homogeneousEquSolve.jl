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
    dy[2] =  M * fR - pE * fL
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