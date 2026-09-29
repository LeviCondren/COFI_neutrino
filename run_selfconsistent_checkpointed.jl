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

const PROJECT_ROOT = "/global/u2/l/lcondren/COFI_neutrino"
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
const MAX_ITER  = parse(Int, get(ENV, "SC_MAX_ITER", "10"))
const SAMPLE_ZS = range(backgroundSolve.z0, backgroundSolve.z1, length=5)
const SAVE_GRID = backgroundSolve.z_grid  # 500 points z0:z1 - used to checkpoint phi_interp itself

# Wall-clock budget: leave a safety margin before SLURM's hard kill so we can
# always finish writing whatever checkpoint is in progress. Must be set by
# the sbatch script to match its own --time (see submit_selfconsistent.sbatch).
#
# This buffer must comfortably exceed the slowest single condensate point
# (~9 min observed) since, unlike a threaded loop, a pmap batch of remaining
# points is only checkpointed once the WHOLE batch returns, not point by
# point - so the deadline check before launching a batch needs enough room
# left for that whole batch (bounded by its slowest point) to finish.
const WALL_SECONDS  = parse(Float64, get(ENV, "SBATCH_WALL_SECONDS", "14400"))
const SAFETY_BUFFER = 900.0
const START_TIME    = time()
const DEADLINE       = START_TIME + WALL_SECONDS - SAFETY_BUFFER

out_of_time() = time() > DEADLINE

phi_checkpoint_path(iter)     = joinpath(CKPT_DIR, "phi_iter_$(iter).csv")
partial_condensate_path(iter) = joinpath(CKPT_DIR, "condensate_partial_iter_$(iter).csv")
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

Each remaining point is independent, so they run as one parallel batch
across worker processes (Distributed.pmap) rather than one at a time.

Wall-clock note: the deadline is only checked once, before launching a
batch - a pmap batch blocks until ALL of its points finish before any of
them are checkpointed (unlike a threaded loop, which could checkpoint each
one as it individually completed). If the job is killed mid-batch, the
whole batch's progress is lost, not just the unfinished points - this is
why SAFETY_BUFFER is set well above the slowest single point's cost, so a
batch launched just before the deadline still has room to finish.
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
        results = pmap(z -> compute_condensate_point(z, phi_zs, phi_vals, P_MAX, RTOL), remaining)

        io = open(partial_condensate_path(iter), "a")
        try
            for (z, val) in zip(remaining, results)
                println("  [iter $iter] z = ", z, "  <psibar psi> = ", val)
                done[z] = val
                println(io, z, ",", val)
                flush(io)
            end
        finally
            close(io)
        end
    end

    vals = [done[z] for z in zgrid]
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
        println("=== pass 0: homogeneous solve, source off ===")
        phi_interp, _ = get_scalar_components(solve_uv_scalar_solution(PE, z -> 0.0))
        save_phi_checkpoint(0, phi_interp)
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

        new_phi_interp, _ = get_scalar_components(solve_uv_scalar_solution(PE, source_interp))
        max_diff = maximum(abs(new_phi_interp(z) - phi_interp(z)) for z in SAMPLE_ZS)
        println("iteration ", iter, ": max change on sample = ", max_diff)

        save_phi_checkpoint(iter, new_phi_interp)
        rm(partial_condensate_path(iter); force=true)  # iteration complete; drop its partial file
        phi_interp = new_phi_interp

        if max_diff < TOL
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
