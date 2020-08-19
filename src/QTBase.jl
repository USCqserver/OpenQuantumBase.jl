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
import StaticArrays: SMatrix, MVector, @MMatrix
import LinearAlgebra.BLAS: her!, gemm!
import SparseArrays: sparse, issparse, spzeros, SparseMatrixCSC
import QuadGK: quadgk

"""
$(TYPEDEF)

Suptertype for Hamiltonians with elements of type `T`. Any Hamiltonian object should implement two interfaces: `H(t)` and `H(du, u, p, t)`.
"""
abstract type AbstractHamiltonian{T <: Number} end


Base.eltype(::AbstractHamiltonian{T}) where {T} = T
Base.size(H::AbstractHamiltonian) = H.size
Base.size(H::AbstractHamiltonian, dim::T) where {T <: Integer} = H.size[dim]
get_cache(H::AbstractHamiltonian) = H.u_cache


"""
$(TYPEDEF)

Base for types defining Hamiltonians using dense matrices.
"""
abstract type AbstractDenseHamiltonian{T <: Number} <: AbstractHamiltonian{T} end


"""
$(TYPEDEF)

Base for types defining Hamiltonians using sparse matrices.
"""
abstract type AbstractSparseHamiltonian{T <: Number} <: AbstractHamiltonian{T} end


"""
$(TYPEDEF)

Base for types defining quantum annealing process.
"""
abstract type AbstractAnnealing{hType <: AbstractHamiltonian,uType <: Union{Vector,Matrix},} end

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
abstract type AbstractLiouvillian end

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

include("hamiltonian/hamiltonian_base.jl")
include("hamiltonian/dense_hamiltonian.jl")
include("hamiltonian/sparse_hamiltonian.jl")
include("hamiltonian/adiabatic_frame_hamiltonian.jl")
include("hamiltonian/interp_hamiltonian.jl")
include("hamiltonian/custom_hamiltonian.jl")

include("annealing/annealing_type.jl")
include("annealing/displays.jl")

include("opensys/opensys_op.jl")
include("opensys/redfield.jl")
include("opensys/davies.jl")
include("opensys/cgme.jl")
include("opensys/stochastic.jl")

include("projection/util.jl")
include("projection/projection.jl")
include("projection/gamma_matrix.jl")

export AbstractHamiltonian, AbstractDenseHamiltonian, AbstractSparseHamiltonian
export AbstractLiouvillian, AbstractCouplings, AbstractTimeDependentCouplings
export AbstractAnnealing, AbstractBath

export temperature_2_β, temperature_2_freq, β_2_temperature, freq_2_temperature
export σx, σz, σy, σi, σ, ⊗, PauliVec, spσx, spσz, spσi, spσy

export q_translate,
    standard_driver, local_field_term, two_local_term, single_clause
export q_translate_state, collective_operator, hamming_weight_operator

export matrix_decompose, check_positivity, check_unitary, partial_trace
export inst_population, gibbs_state, low_level_matrix, ame_jump
export construct_interpolations, gradient, log_uniform

export SparseHamiltonian, DenseHamiltonian, AdiabaticFrameHamiltonian
export InterpDenseHamiltonian, InterpSparseHamiltonian, CustomDenseHamiltonian
export hamiltonian_from_function,
    evaluate, issparse, get_cache, update_cache!, update_vectorized_cache!

export ConstantCouplings, TimeDependentCoupling
export TimeDependentCouplings, CustomCouplings, collective_coupling

export eigen_decomp, EIGEN_DEFAULT

export Annealing, ODEParams, Interaction, InteractionSet, set_u0!

export RedfieldGenerator, DaviesGenerator, Fluctuators
export OpenSysOp, AMEOperator, FluctuatorOperator, RedfieldOperator

export correlation, polaron_correlation, γ, S, spectrum, info_freq
export Ohmic, OhmicBath, EnsembleFluctuator, CustomBath, CorrelatedBath

export τ_B, τ_SB, coarse_grain_timescale
export build_redfield, build_davies, build_CGG, build_fluctuator

export InplaceUnitary

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