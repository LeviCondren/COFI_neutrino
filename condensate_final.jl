# One-shot condensate profile <psibar psi>(z) for the FINAL phi of a finished
# self-consistent run (see run_selfconsistent_checkpointed.jl), computed as
# accurately as is affordable once rather than per iteration:
#
#   - phi is the exact KG ODE solution (dense output), rebuilt from the final
#     iteration's source condensate_iter_N.csv - not the linear interpolant of
#     the 500-point phi CSV.
#   - The UV/IR Dirac (angle-form) solutions do not depend on z, so for each
#     momentum p one solve pair gives the trace on the whole z grid; a single
#     vector-valued p-integral then yields the condensate at every z at once.
#   - Pass 1 gets a rough profile; pass 2 reruns the integral with a norm
#     weighted by 1/|pass-1 value| so the tolerance is RELATIVE at every z,
#     not relative to the largest entry.
#
# Tunables (env): CF_N_Z (z points), CF_RTOL (relative tolerance per z),
# CF_P_MAX (momentum cutoff), SC_CKPT_DIR, SC_PE.

using DelimitedFiles, Interpolations, QuadGK, Printf, Plots
include(joinpath(@__DIR__, "backgroundSolve.jl"))
using .backgroundSolve
const B_ = backgroundSolve

const CKPT_DIR = get(ENV, "SC_CKPT_DIR", joinpath(@__DIR__, "checkpoints"))
const PE       = parse(Float64, get(ENV, "SC_PE", "1.0"))
const N_Z      = parse(Int, get(ENV, "CF_N_Z", "1000"))
const RTOL     = parse(Float64, get(ENV, "CF_RTOL", "1e-8"))
const P_MAX    = parse(Float64, get(ENV, "CF_P_MAX", string(B_.p_max)))
const ZERO_PHI = z -> 0.0

pr(x...) = (println(x...); flush(stdout))

function final_iteration()
    for line in eachline(joinpath(CKPT_DIR, "state.txt"))
        startswith(line, "last_completed_iteration=") && return parse(Int, split(line, "=")[2])
    end
    error("no last_completed_iteration in state.txt")
end

"""Rebuild the final phi exactly: the KG UV solve sourced by the linear
interpolant of condensate_iter_N.csv, same as the driver did."""
function rebuild_final_phi(iter)
    src = readdlm(joinpath(CKPT_DIR, "condensate_iter_$(iter).csv"), ',')
    source = linear_interpolation(src[:, 1], src[:, 2]; extrapolation_bc=Line())
    sol = solve_uv_scalar_solution(PE, source)
    phi = z -> sol(z; idxs=1)
    saved = readdlm(joinpath(CKPT_DIR, "phi_iter_$(iter).csv"), ',')
    mismatch = maximum(abs(phi(z) - v) for (z, v) in eachrow(saved)) / maximum(abs, saved[:, 2])
    pr("rebuilt phi_iter_$iter: max |rebuilt - saved CSV| / max|phi| = ", mismatch)
    mismatch < 1e-10 || @warn "rebuilt phi does not match the saved checkpoint" mismatch
    return phi
end

"""Chirality-mixing trace at every z in zs for momentum p (cf. propagator_trace)."""
function trace_on_grid(p, phi, zs)
    u = solve_uv_angle(p, phi)
    v = solve_ir_angle(p, phi)
    return [(θu = u(z); θv = v(z); -cos(θu + θv) / sin(θv - θu)) for z in zs]
end

function condensate_profile(phi, zs)
    cache = Dict{Float64, Vector{Float64}}()   # pass 2 reuses pass-1 nodes
    nevals = Ref(0)
    integrand(p) = get!(cache, p) do
        nevals[] += 1
        (p^3 / (8π^2)) .* (trace_on_grid(p, phi, zs) .- trace_on_grid(p, ZERO_PHI, zs))
    end

    t = @elapsed I1, E1 = quadgk(integrand, 0.0, P_MAX; rtol=1e-4)
    pr(@sprintf("pass 1: rtol 1e-4 (max-norm), %d integrand evals, %.1f s", nevals[], t))

    w = abs.(I1) .+ 1e-12 * maximum(abs, I1)
    wnorm(x) = maximum(abs.(x) ./ w)
    t = @elapsed I, E = quadgk(integrand, 0.0, P_MAX; rtol=RTOL, norm=wnorm)
    pr(@sprintf("pass 2: rtol %.0e relative per z, %d integrand evals total, %.1f s, est. max relative error %.2e",
                RTOL, nevals[], t, E / wnorm(I)))
    return I, E / wnorm(I)
end

function main()
    iter = final_iteration()
    flag = isfile(joinpath(CKPT_DIR, "CONVERGED")) ? "CONVERGED" :
           isfile(joinpath(CKPT_DIR, "MAX_ITER_REACHED")) ? "MAX_ITER_REACHED" : "UNFINISHED"
    pr("final iteration $iter ($flag); N_Z = $N_Z, rtol = $RTOL, p_max = $P_MAX")
    phi = rebuild_final_phi(iter)

    zs = collect(range(B_.z0, B_.z1, length=N_Z))
    vals, relerr = condensate_profile(phi, zs)

    out = joinpath(CKPT_DIR, "condensate_final.csv")
    open(out * ".tmp", "w") do io
        println(io, "# exact condensate for phi_iter_$iter ($flag); rtol=$RTOL per z, est. max rel err=$relerr, p_max=$P_MAX")
        for (z, v) in zip(zs, vals)
            println(io, z, ",", v)
        end
    end
    mv(out * ".tmp", out; force=true)
    pr("wrote ", out)

    # Spot check against the independent per-z routine used by the driver.
    for z in (B_.z0 + 0.25 * (B_.z1 - B_.z0), B_.z1 - 0.1 * (B_.z1 - B_.z0))
        i = argmin(abs.(zs .- z))
        ref = renormalized_condensate(zs[i], phi, P_MAX; rtol=RTOL)
        pr(@sprintf("check z=%.3f: profile %.12g  per-z quadgk %.12g  rel diff %.1e", zs[i], vals[i], ref, abs(vals[i] - ref) / abs(ref)))
    end

    plt = plot(zs, vals; xlabel="z", ylabel="<psibar psi>(z)", legend=false, linewidth=2,
               title="Condensate for final phi (iteration $iter, $flag)")
    savefig(plt, joinpath(CKPT_DIR, "condensate_final.png"))
end

main()
