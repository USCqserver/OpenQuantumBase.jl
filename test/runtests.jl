using SafeTestsets

@time begin

@time @safetestset "Math Utilities" begin include("math_util.jl") end
@time @safetestset "Interpolations" begin include("interpolations.jl") end
@time @safetestset "Multi-qubits Hamiltonian Construction" begin include("multi_hamiltonian.jl") end
@time @safetestset "Dense Hamiltonian" begin include("dense_hamiltonian.jl") end
@time @safetestset "Sparse Hamiltonian" begin include("sparse_hamiltonian.jl") end
@time @safetestset "Adiabatic Frame Hamiltonian" begin include("adiabatic_frame_hamiltonian.jl") end
# this test will take a long time and is currently broken
#@time @safetestset "Differential Equations" begin include("diff_eq.jl") end
end
