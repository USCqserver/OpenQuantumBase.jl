using SafeTestsets

@time begin

    @time @safetestset "Math Utilities" begin
        include("math_util.jl")
    end
    @time @safetestset "Interpolations" begin
        include("interpolations.jl")
    end
    @time @safetestset "Multi-qubits Hamiltonian Construction" begin
        include("multi_hamiltonian.jl")
    end
    @time @safetestset "Dense Hamiltonian" begin
        include("hamiltonian/dense_hamiltonian.jl")
    end
    @time @safetestset "Sparse Hamiltonian" begin
        include("hamiltonian/sparse_hamiltonian.jl")
    end
    @time @safetestset "Adiabatic Frame Hamiltonian" begin
        include("hamiltonian/adiabatic_frame_hamiltonian.jl")
    end
    @time @safetestset "Interpolation Hamiltonian" begin
        include("hamiltonian/interp_hamiltonian.jl")
    end
    @time @safetestset "Coupling" begin
        include("coupling.jl")
    end
    @time @safetestset "Davies and AME" begin
        include("davies.jl")
    end
    @time @safetestset "Redfield" begin
        include("redfield.jl")
    end
    @time @safetestset "Annealing" begin
        include("annealing.jl")
    end
    @time @safetestset "Projections" begin
        include("projection.jl")
    end
    @time @safetestset "Γ Matrices" begin
        include("gamma_matrix.jl")
    end
end
