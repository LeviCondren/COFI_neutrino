# Resumable driver for the self-consistent scalar-fermion background solve,
# meant to be run repeatedly from a self-resubmitting sbatch job chain (see
# submit_selfconsistent.sbatch). Each invocation:
#
#   - resumes from the last completed iteration's checkpoint, if any
#   - checkpoints every single condensate tabulation point as it's computed,
#     not just at iteration boundaries - a single tabulation pass (n_points
#     calls to renormalized_condensate) can itself take far longer than one
#     job's wall-clock budget
#   - stops once converged (or once max_iter is reached without converging),
#     writes a flag file so the sbatch wrapper knows to stop resubmitting,
#     records the final phi(z), and saves a plot
#
# All tunables below can be overridden via environment variables, so the
# same script serves both real runs and quick logic tests.

# Process-based parallelism (Distributed.jl), NOT threads: the
# DifferentialEquations.jl/QuadGK call chain inside renormalized_condensate
# is not safe to call concurrently from multiple threads sharing one
# process - confirmed by testing, it throws
# `ConcurrencyViolationError("Vector can not be resized concurrently")`.
# Separate worker processes have no shared memory to corrupt, so they sidestep
# that whole class of bug.
using Distributed

const PROJECT_ROOT = @__DIR__
const N_WORKERS = parse(Int, get(ENV, "SC_NWORKERS", get(ENV, "SLURM_CPUS_PER_TASK", "4")))
addprocs(N_WORKERS)

@everywhere include(joinpath($PROJECT_ROOT, "backgroundSolve.jl"))
@everywhere using .backgroundSolve
@everywhere using Interpolations

# The per-point work, defined on every worker (and the master). phi is
# shipped as plain (zs, vals) arrays rather than the ODE-solution-backed
# closure itself, so what crosses process boundaries is simple and reliably
# serializable.
@everywhere function compute_condensate_point(z, phi_zs, phi_vals, p_max, rtol)
    local_phi = linear_interpolation(phi_zs, phi_vals; extrapolation_bc=Line())
    return renormalized_condensate(z, local_phi, p_max; rtol=rtol)
end

using Plots

const CKPT_DIR = get(ENV, "SC_CKPT_DIR", "checkpoints")
mkpath(CKPT_DIR)

const PE        = parse(Float64, get(ENV, "SC_PE", "1.0"))
const P_MAX     = backgroundSolve.p_max
const N_POINTS  = parse(Int, get(ENV, "SC_N_POINTS", "12"))
const RTOL      = parse(Float64, get(ENV, "SC_RTOL", "1e-3"))
const TOL       = parse(Float64, get(ENV, "SC_TOL", "1e-4"))
# If > 0, converge on the RELATIVE change max|Δphi| / max|phi| < REL_TOL instead
# of the absolute TOL (phi can reach ~1e7, making any absolute tolerance arbitrary).
const REL_TOL   = parse(Float64, get(ENV, "SC_REL_TOL", "0"))
const MAX_ITER  = parse(Int, get(ENV, "SC_MAX_ITER", "10"))
const SAMPLE_ZS = range(backgroundSolve.z0, backgroundSolve.z1, length=5)
const SAVE_GRID = backgroundSolve.z_grid  # 500 points z0:z1 - used to checkpoint phi_interp itself
# Pass-0 profile: "homogeneous" = UV solve of the KG equation with the source off;
# "ansatz" = power law A z^Δ- + B z^Δ+ (see power_law_ansatz).
const INIT      = get(ENV, "SC_INIT", "homogeneous")

# Wall-clock budget: leave a safety margin before SLURM's hard kill so we can
# always finish writing whatever checkpoint is in progress. Must be set by
# the sbatch script to match its own --time (see submit_selfconsistent.sbatch).
#
# No new condensate point is started after DEADLINE. Points are checkpointed
# individually as they finish, so the buffer should exceed the slowest single
# point (up to ~35 min observed on HPC3) for in-flight points to complete;
# any that don't are simply recomputed by the next job.
const WALL_SECONDS  = parse(Float64, get(ENV, "SBATCH_WALL_SECONDS", "14400"))
const SAFETY_BUFFER = parse(Float64, get(ENV, "SC_SAFETY_BUFFER", "900"))
const START_TIME    = time()
const DEADLINE       = START_TIME + WALL_SECONDS - SAFETY_BUFFER

out_of_time() = time() > DEADLINE

"""
Homogeneous solution of the KG equation, z^2 phi'' - 3 z phi' - m_phi^2 phi = 0:
phi = A z^Δ- + B z^Δ+ with Δ± = 2 ± sqrt(4 + m_phi^2), A and B fixed by the same
UV conditions as solve_uv_scalar_solution, phi(z0) = 1 and phi'(z0) = 0.
"""
function power_law_ansatz()
    m, z0 = backgroundSolve.m_phi, backgroundSolve.z0
    dm, dp = 2 - sqrt(4 + m^2), 2 + sqrt(4 + m^2)
    A, B = [z0^dm z0^dp; dm*z0^(dm-1) dp*z0^(dp-1)] \ [1.0, 0.0]
    println("power-law ansatz: Δ- = $dm, Δ+ = $dp, A = $A, B = $B")
    return z -> A * z^dm + B * z^dp
end

phi_checkpoint_path(iter)     = joinpath(CKPT_DIR, "phi_iter_$(iter).csv")
partial_condensate_path(iter) = joinpath(CKPT_DIR, "condensate_partial_iter_$(iter).csv")
# Complete, z-sorted source of iteration iter: phi_iter_iter is exactly the KG solve
# sourced by the linear interpolant of these points (used by condensate_final.jl).
condensate_path(iter)         = joinpath(CKPT_DIR, "condensate_iter_$(iter).csv")
state_path()      = joinpath(CKPT_DIR, "state.txt")
converged_path()  = joinpath(CKPT_DIR, "CONVERGED")
maxiter_path()    = joinpath(CKPT_DIR, "MAX_ITER_REACHED")
final_csv_path()  = joinpath(CKPT_DIR, "final_phi.csv")
final_plot_path() = joinpath(CKPT_DIR, "final_phi_plot.png")

"""Write via a temp file + atomic rename, so a job killed mid-write never
leaves a torn checkpoint that a later resume would read as valid."""
function atomic_write(f::Function, path::String)
    tmp = path * ".tmp"
    open(f, tmp, "w")
    mv(tmp, path; force=true)
end

function save_phi_checkpoint(iter, phi_interp)
    atomic_write(phi_checkpoint_path(iter)) do io
        for z in SAVE_GRID
            println(io, z, ",", phi_interp(z))
        end
    end
    atomic_write(state_path()) do io
        println(io, "last_completed_iteration=", iter)
    end
end

function load_phi_checkpoint(iter)
    zs = Float64[]; vals = Float64[]
    for line in eachline(phi_checkpoint_path(iter))
        parts = split(line, ",")
        length(parts) == 2 || continue  # skip any torn trailing line, just in case
        push!(zs, parse(Float64, parts[1]))
        push!(vals, parse(Float64, parts[2]))
    end
    return linear_interpolation(zs, vals; extrapolation_bc=Line())
end

function last_completed_iteration()
    isfile(state_path()) || return nothing
    for line in eachline(state_path())
        if startswith(line, "last_completed_iteration=")
            return parse(Int, split(line, "=")[2])
        end
    end
    return nothing
end

function load_partial_condensate(iter)
    zs = Float64[]; vals = Float64[]
    path = partial_condensate_path(iter)
    if isfile(path)
        for line in eachline(path)
            parts = split(line, ",")
            length(parts) == 2 || continue  # skip a torn trailing line from a killed write
            push!(zs, parse(Float64, parts[1]))
            push!(vals, parse(Float64, parts[2]))
        end
    end
    return zs, vals
end

"""
Tabulate the condensate for one iteration, checkpointing every point as it's
computed and resuming any points a previous (killed) run already finished.
Returns `nothing` if we ran out of wall-clock time before finishing all
n_points - in that case, everything computed so far is already safely on
disk for the next resubmission to pick up.

Each remaining point is independent. One task per worker process pulls
points off a shared queue and ships each to its worker with remotecall_fetch;
each result is appended to the partial file as soon as it returns, so a job
killed mid-iteration (wall time or preemption) loses only its in-flight
points. The deadline is checked before starting each point.
"""
function tabulate_condensate_resumable(phi_interp, iter)
    zgrid = collect(range(backgroundSolve.z0, backgroundSolve.z1, length=N_POINTS))
    done_zs, done_vals = load_partial_condensate(iter)
    done = Dict(zip(done_zs, done_vals))

    remaining = [z for z in zgrid if !haskey(done, z)]

    if !isempty(remaining)
        if out_of_time()
            println("Out of wall-clock time; checkpointed ", length(done), "/", N_POINTS,
                    " condensate points for iteration ", iter, ". Exiting for resubmission.")
            return nothing
        end

        println("Computing ", length(remaining), " remaining condensate point(s) across ",
                nworkers(), " worker process(es) ...")

        phi_zs = collect(SAVE_GRID)
        phi_vals = phi_interp.(phi_zs)
        queue = Channel{Float64}(length(remaining))
        foreach(z -> put!(queue, z), remaining)
        close(queue)

        # Master-side tasks are cooperatively scheduled on one thread, so the
        # shared io/done updates between yield points don't race.
        io = open(partial_condensate_path(iter), "a")
        try
            @sync for w in workers()
                @async for z in queue
                    out_of_time() && break
                    val = remotecall_fetch(compute_condensate_point, w, z, phi_zs, phi_vals, P_MAX, RTOL)
                    done[z] = val
                    println(io, z, ",", val); flush(io)
                    println("  [iter $iter] z = ", z, "  <psibar psi> = ", val,
                            "  (", length(done), "/", N_POINTS, ")"); flush(stdout)
                end
            end
        finally
            close(io)
        end

        if length(done) < N_POINTS
            println("Out of wall-clock time; checkpointed ", length(done), "/", N_POINTS,
                    " condensate points for iteration ", iter, ". Exiting for resubmission.")
            return nothing
        end
    end

    vals = [done[z] for z in zgrid]
    atomic_write(condensate_path(iter)) do io
        for (z, v) in zip(zgrid, vals)
            println(io, z, ",", v)
        end
    end
    return linear_interpolation(zgrid, vals; extrapolation_bc=Line())
end

function make_final_plot(phi_interp, iter)
    zs = range(backgroundSolve.z0, backgroundSolve.z1, length=500)
    plt = plot(zs, phi_interp.(zs),
               xlabel="z", ylabel="phi(z)",
               title="Self-consistent scalar background (iteration $iter)",
               legend=false, linewidth=2)
    savefig(plt, final_plot_path())
end

function finish!(phi_interp, iter, flag_path, message)
    println(message)
    touch(flag_path)
    cp(phi_checkpoint_path(iter), final_csv_path(); force=true)
    make_final_plot(phi_interp, iter)
end

function main()
    last_iter = last_completed_iteration()

    local phi_interp
    if last_iter === nothing
        if INIT == "ansatz"
            println("=== pass 0: power-law ansatz ===")
            phi0 = power_law_ansatz()
        elseif INIT == "homogeneous"
            println("=== pass 0: homogeneous solve, source off ===")
            phi0, _ = get_scalar_components(solve_uv_scalar_solution(PE, z -> 0.0))
        else
            error("unknown SC_INIT = $INIT (expected \"homogeneous\" or \"ansatz\")")
        end
        save_phi_checkpoint(0, phi0)
        phi_interp = load_phi_checkpoint(0)
        last_iter = 0
    else
        println("Resuming from checkpointed iteration ", last_iter)
        phi_interp = load_phi_checkpoint(last_iter)
    end

    for iter in (last_iter + 1):MAX_ITER
        if out_of_time()
            println("Out of wall-clock time before starting iteration ", iter, "; exiting for resubmission.")
            return
        end

        println("=== self-consistency iteration ", iter, " (source on) ===")
        source_interp = tabulate_condensate_resumable(phi_interp, iter)
        source_interp === nothing && return  # already checkpointed per-point above

        new_phi, _ = get_scalar_components(solve_uv_scalar_solution(PE, source_interp))
        save_phi_checkpoint(iter, new_phi)
        # Compare checkpoint-to-checkpoint (both linear on SAVE_GRID). Comparing
        # the raw ODE solution against a checkpoint-loaded phi instead measures
        # linear-interpolation error, which is O(1) here since phi swings by
        # hundreds per grid step - that spuriously blocked convergence after a resume.
        new_phi_interp = load_phi_checkpoint(iter)
        max_diff = maximum(abs(new_phi_interp(z) - phi_interp(z)) for z in SAMPLE_ZS)
        rel_diff = max_diff / maximum(abs, new_phi_interp.(SAVE_GRID))
        println("iteration ", iter, ": max change on sample = ", max_diff,
                "  (relative to max|phi|: ", rel_diff, ")"); flush(stdout)
        converged = REL_TOL > 0 ? rel_diff < REL_TOL : max_diff < TOL

        rm(partial_condensate_path(iter); force=true)  # iteration complete; drop its partial file
        phi_interp = new_phi_interp

        if converged
            finish!(phi_interp, iter, converged_path(), "Converged after $iter iterations.")
            return
        end

        if iter == MAX_ITER
            finish!(phi_interp, iter, maxiter_path(), "Reached max_iter ($MAX_ITER) without converging.")
            return
        end
    end
end

main()
