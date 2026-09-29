import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

# ==================================================================
# Physical parameters (module-level, matching your Mathematica setup)
# ==================================================================

k = 1.0
z0 = 1.0
z1 = 100.0

c_psi = 1.0
m_psi = c_psi * k
y5 = 0.1

# Grid
N_z = 200
z_grid = np.linspace(z0, z1, N_z)


# ==================================================================
# Solvers
# ==================================================================

def solve_uv_solution(pE, phi_interp, rtol=1e-8, atol=1e-10):
    """
    Solve the coupled Dirac ODEs with UV boundary conditions:
        fR(z0) = 0  (from Psi_R Dirichlet at UV brane)
        fL(z0) = 1  (normalization)
    
    Integrates from z0 to z1.
    
    Parameters
    ----------
    pE : float
        Momentum eigenvalue (Euclidean).
    phi_interp : callable
        Function returning <Phi(z)> for scalar z in [z0, z1].
    rtol, atol : float
        Relative and absolute tolerances for the ODE solver.
    
    Returns
    -------
    fL_func, fR_func : callables
        Interpolating functions for f_L(z) and f_R(z).
    """
    def M_func(z):
        return (m_psi + y5 * phi_interp(z)) / (k * z)
    
    def rhs(z, y):
        # y = [fL, fR]
        fL, fR = y
        M = M_func(z)
        dfL = -M * fL + pE * fR
        dfR =  M * fR - pE * fL
        return [dfL, dfR]
    
    # Initial conditions at z0: fL = 1, fR = 0
    y0 = [1.0, 0.0]
    
    sol = solve_ivp(
        rhs, 
        [z0, z1], 
        y0,
        method='DOP853',       # High-order Runge-Kutta; robust default
        dense_output=True,     # Enables continuous interpolation
        rtol=rtol, 
        atol=atol
    )
    
    if not sol.success:
        raise RuntimeError(f"UV ODE integration failed: {sol.message}")
    
    # sol.sol is a callable returning [fL(z), fR(z)]
    fL_func = lambda z: sol.sol(z)[0]
    fR_func = lambda z: sol.sol(z)[1]
    
    return fL_func, fR_func


def solve_ir_solution(pE, phi_interp, rtol=1e-8, atol=1e-10):
    """
    Solve the coupled Dirac ODEs with IR boundary conditions:
        fL(z1) = 0  (from Psi_L Dirichlet at IR brane)
        fR(z1) = 1  (normalization)
    
    Integrates BACKWARD from z1 to z0.
    """
    def M_func(z):
        return (m_psi + y5 * phi_interp(z)) / (k * z)
    
    def rhs(z, y):
        fL, fR = y
        M = M_func(z)
        dfL = -M * fL + pE * fR
        dfR =  M * fR - pE * fL
        return [dfL, dfR]
    
    # Initial conditions at z1: fL = 0, fR = 1
    y0 = [0.0, 1.0]
    
    # Integrate backward: from z1 down to z0
    sol = solve_ivp(
        rhs, 
        [z1, z0],   # Backward integration
        y0,
        method='DOP853',
        dense_output=True,
        rtol=rtol, 
        atol=atol
    )
    
    if not sol.success:
        raise RuntimeError(f"IR ODE integration failed: {sol.message}")
    
    fL_func = lambda z: sol.sol(z)[0]
    fR_func = lambda z: sol.sol(z)[1]
    
    return fL_func, fR_func