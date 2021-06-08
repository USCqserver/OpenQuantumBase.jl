#using Test: include
using SafeTestsets

@time begin
    @time @safetestset "Base Utilities" begin
        include("utilities/base_util.jl")
    end
    @time @safetestset "Math Utilities" begin
        include("utilities/math_util.jl")
    end
    @time @safetestset "Multi-qubits Hamiltonian Construction" begin
        include("utilities/multi_qubits_construction.jl")
    end
    @time @safetestset "Development tools" begin
        include("dev_tools.jl")
    end
    @time @safetestset "Displays" begin
        include("utilities/display.jl")
    end
    @time @safetestset "Interpolations" begin
        include("interpolations.jl")
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
    @time @safetestset "Custom Hamiltonian" begin
        include("hamiltonian/custom_hamiltonian.jl")
    end
    @time @safetestset "Coupling" begin
        include("coupling_bath_interaction/coupling.jl")
    end
    @time @safetestset "Bath" begin
        include("coupling_bath_interaction/bath.jl")
    end
    @time @safetestset "Interactions" begin
        include("coupling_bath_interaction/interaction.jl")
    end
    @time @safetestset "Davies and AME" begin
        include("opensys/davies.jl")
    end
    @time @safetestset "Redfield" begin
        include("opensys/redfield.jl")
    end
    @time @safetestset "Lindblad" begin
        include("opensys/lindblad.jl")
    end
    @time @safetestset "Stochastic" begin
        include("opensys/stochastic.jl")
    end
    @time @safetestset "Annealing/Evolution" begin
        include("annealing.jl")
    end
    @time @safetestset "Projections" begin
        include("projection.jl")
    end
end
