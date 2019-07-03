using SafeTestsets

@time begin

@time @safetestset "Math Utilities" begin include("math_util.jl") end
@time @safetestset "Ising Hamiltonian Construction" begin include("ising_hamiltonian.jl") end
@time @safetestset "Time Dependent Hamiltonian" begin include("time_dependent_hamiltonian.jl") end
@time @safetestset "Adiabatic Frame Hamiltonian" begin include("adiabatic_frame_hamiltonian.jl") end

end
