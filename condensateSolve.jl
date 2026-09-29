using QuadGK

# ==================================================================
# Trace of the propagator
# ==================================================================

function propagator_trace(pE, phi_interp, z)
    UVsol = solve_uv_solution(pE, phi_interp)
    IRsol = solve_ir_solution(pE, phi_interp)
    fL_uv, fR_uv = get_components(UVsol)
    fL_ir, fR_ir = get_components(IRsol)
    
    # Wronskian: evaluate at fixed midpoint since it's z-independent
    z_mid = (z0 + z1) / 2
    W = fL_uv(z_mid) * fR_ir(z_mid) - fR_uv(z_mid) * fL_ir(z_mid)
    
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