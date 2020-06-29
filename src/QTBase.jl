module QTBase

using DocStringExtensions

import LinearAlgebra:
    mul!,
    axpy!,
    I,
    ishermitian,
    Hermitian,
    eigen,
    tr,
    eigen!,
    diag,
    lmul!,
    Diagonal
import LinearAlgebra.BLAS: her!, gemm!
import SparseArrays: sparse, issparse, spzeros, SparseMatrixCSC
import QuadGK: quadgk

"""
$(TYPEDEF)

Suptertype for Hamiltonians with elements of type `T`. Any Hamiltonian object should implement two interfaces: `H(t)` and `H(du, u, p, t)`.
"""
abstract type AbstractHamiltonian{T<:Number} end


Base.eltype(::AbstractHamiltonian{T}) where {T} = T
Base.size(H::AbstractHamiltonian) = H.size
Base.size(H::AbstractHamiltonian, dim::T) where {T<:Integer} = H.size[dim]
get_cache(H::AbstractHamiltonian) = H.u_cache


"""
$(TYPEDEF)

Base for types defining Hamiltonians using dense matrices.
"""
abstract type AbstractDenseHamiltonian{T<:Number} <: AbstractHamiltonian{T} end


"""
$(TYPEDEF)

Base for types defining Hamiltonians using sparse matrices.
"""
abstract type AbstractSparseHamiltonian{T<:Number} <: AbstractHamiltonian{T} end


"""
$(TYPEDEF)

Base for types defining quantum annealing process.
"""
abstract type AbstractAnnealing{
    hType<:AbstractHamiltonian,
    uType<:Union{Vector,Matrix},
} end


"""
$(TYPEDEF)

Base for types defining various control protocols in quantum annealing process.
"""
abstract type AbstractAnnealingControl end

"""
$(TYPEDEF)

Base for types defining system bath coupling operators in open quantum system models.
"""
abstract type AbstractCouplings end

"""
$(TYPEDEF)

Base for types defining bath object.
"""
abstract type AbstractBath end

"""
$(TYPEDEF)
"""
abstract type AbstractOpenSys end

include("base_util.jl")
include("unit_util.jl")
include("math_util.jl")
include("matrix_util.jl")
include("interpolation.jl")

include("coupling/coupling.jl")
include("coupling/interaction.jl")

include("bath/bath_util.jl")
include("bath/ohmic.jl")
include("bath/spin_fluc.jl")
include("bath/custom.jl")

include("integration/cpvagk.jl")
include("integration/ext_util.jl")

include("hamiltonian/dense_hamiltonian.jl")
include("hamiltonian/sparse_hamiltonian.jl")
include("hamiltonian/adiabatic_frame_hamiltonian.jl")
include("hamiltonian/interp_hamiltonian.jl")
include("hamiltonian/custom_hamiltonian.jl")
include("hamiltonian/util.jl")

include("annealing/annealing_type.jl")
include("annealing/displays.jl")

include("opensys/redfield.jl")
include("opensys/davies.jl")
include("opensys/cgme.jl")
include("opensys/opensys_set.jl")

include("projection/util.jl")
include("projection/projection.jl")
include("projection/gamma_matrix.jl")



export temperature_2_beta,
    temperature_2_freq, beta_2_temperature, freq_2_temperature

export σx, σz, σy, σi, σ, ⊗, PauliVec, spσx, spσz, spσi, spσy

export q_translate,
    construct_hamming_weight_op,
    single_clause,
    standard_driver,
    collective_operator,
    GHZ_entanglement_witness,
    local_field_term,
    two_local_term,
    q_translate_state

export matrix_decompose, check_positivity, check_unitary

export construct_interpolations, gradient

export inst_population, gibbs_state, low_level_matrix, ame_jump, update_ρ!

export AbstractHamiltonian,
    AbstractSparseHamiltonian,
    SparseHamiltonian,
    AbstractDenseHamiltonian,
    DenseHamiltonian,
    AdiabaticFrameHamiltonian,
    InterpDenseHamiltonian,
    InterpSparseHamiltonian,
    CustomDenseHamiltonian,
    hamiltonian_from_function,
    evaluate,
    to_dense,
    to_sparse,
    is_sparse,
    get_cache,
    update_cache!,
    update_vectorized_cache!

export AbstractCouplings,
    ConstantCouplings,
    TimeDependentCoupling,
    AbstractTimeDependentCouplings,
    TimeDependentCouplings,
    CustomCouplings,
    collective_coupling

export eigen_decomp, EIGEN_DEFAULT

export AbstractAnnealing,
    Annealing, ODEParams, AbstractAnnealingControl, Interaction, InteractionSet

export AbstractBath,
    AbstractOpenSys,
    OpenSysSet,
    Redfield,
    CGOP,
    RedfieldSet,
    DaviesGenerator,
    AMEDiffEqOperator,
    AMETrajectoryOperator

export τ_B,
    τ_SB,
    coarse_grain_timescale,
    correlation,
    γ,
    S,
    spectrum,
    Ohmic,
    OhmicBath,
    CustomBath,
    EnsembleFluctuator,
    construct_distribution,
    build_redfield,
    build_davies,
    build_CGME

export UnitTime, InplaceUnitary

export ProjectedSystem,
    ProjectedTG,
    project_to_lowlevel,
    get_dθ,
    concatenate,
    ProjectedCoupling,
    construct_projected_coupling,
    construct_projected_TG,
    landau_zener_rotate_angle,
    landau_zener_rotate,
    ΓMatrix

end  # module QTBase
