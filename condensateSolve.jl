using QuadGK

# ==================================================================
# Trace of the propagator
# ==================================================================

"""
Chirality-mixing trace (G_LR + G_RL) at z, computed from the Prüfer angles
θ_uv, θ_ir of the UV/IR solutions (see solve_uv_angle). Identical in exact
arithmetic to propagator_trace_direct (numerator and W both at z, so the
(kz)^2 scalings of get_components cancel), but cannot overflow:

    (-fL_uv fL_ir + fR_uv fR_ir) / W  =  -cos(θ_uv + θ_ir) / sin(θ_ir - θ_uv)
"""
function propagator_trace(pE, phi_interp, z)
    θu = solve_uv_angle(pE, phi_interp)(z)
    θv = solve_ir_angle(pE, phi_interp)(z)
    return -cos(θu + θv) / sin(θv - θu)
end

# Original direct (fL, fR) form - overflows for large |phi|. Kept for cross-checks.
function propagator_trace_direct(pE, phi_interp, z)
    UVsol = solve_uv_solution(pE, phi_interp)
    IRsol = solve_ir_solution(pE, phi_interp)
    fL_uv, fR_uv = get_components(UVsol)
    fL_ir, fR_ir = get_components(IRsol)
    
    # Wronskian at the same z as the numerator: the unscaled W is z-independent,
    # but get_components' (kz)^2 scaling makes the scaled one go like z^4, so
    # taking it at a fixed z_mid would leave an arbitrary (z/z_mid)^4 factor.
    W = fL_uv(z) * fR_ir(z) - fR_uv(z) * fL_ir(z)
    
    # Chirality-mixing components (for bar-Psi Psi condensate)
    # G_LR + G_RL ~ -fL_uv * fL_ir + fR_uv * fR_ir
    trace = (-fL_uv(z) * fL_ir(z) + fR_uv(z) * fR_ir(z)) / W
    
    return trace
end

function renormalized_condensate(z, phi_interp, p_max; rtol=1e-6)
    integrand_full(p) = (p^3 / (8*π^2)) * propagator_trace(p, phi_interp, z)
    integrand_ref(p)  = (p^3 / (8*π^2)) * propagator_trace(p, z -> 0.0, z)
    
    integrand_diff(p) = integrand_full(p) - integrand_ref(p)
    
    integral, _ = quadgk(integrand_diff, 0.0, p_max; rtol=rtol)
    return integral
end