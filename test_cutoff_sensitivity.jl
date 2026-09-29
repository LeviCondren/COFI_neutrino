include("backgroundSolve.jl")
using .backgroundSolve: renormalized_condensate

for p_max in [5.0, 10.0, 20.0, 25.0]
    phi_interp = z -> 0.1 * exp(-z / 10.0)  # Example interpolating function
    result = renormalized_condensate(10.0, phi_interp; p_max=p_max)
    println("p_max = $p_max: condensate = $result")
end