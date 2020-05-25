module QTBase

using DocStringExtensions

import LinearAlgebra: kron,
                      mul!,
                      axpy!,
                      I,
                      ishermitian,
                      Hermitian,
                      eigmin,
                      eigen,
                      tr,
                      eigen!,
                      axpy!,
                      diag,
                      lmul!,
                      Diagonal,
                      normalize
import LinearAlgebra.BLAS: her!, gemm!
import SparseArrays: sparse, issparse, spzeros, SparseMatrixCSC
import Arpack: eigs
import QuadGK: quadgk
import Interpolations: interpolate,
                       BSpline,
                       Quadratic,
                       Line,
                       OnGrid,
                       scale,
                       gradient1,
                       extrapolate,
                       Linear,
                       Gridded,
                       NoInterp,
                       Cubic,
                       Constant
import StatsBase: sample, Weights


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


"""
$(TYPEDEF)
"""
abstract type OpenSysSets <: AbstractOpenSys end

include("utils.jl")
include("unit_util.jl")
include("math_util.jl")
include("matrix_util.jl")
include("interpolation.jl")
include("coupling.jl")


include("integration/cpvagk.jl")


include("hamiltonian/dense_hamiltonian.jl")
include("hamiltonian/sparse_hamiltonian.jl")
include("hamiltonian/adiabatic_frame_hamiltonian.jl")
include("hamiltonian/interp_hamiltonian.jl")
include("hamiltonian/custom_hamiltonian.jl")
include("hamiltonian/util.jl")


include("opensys/redfield.jl")
include("opensys/davies.jl")


include("annealing/annealing_type.jl")
#include("annealing/annealing_params.jl")
include("annealing/displays.jl")

include("projection/util.jl")
include("projection/projection.jl")
include("projection/gamma_matrix.jl")



export temperature_2_beta,
       temperature_2_freq,
       beta_2_temperature,
       freq_2_temperature

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

export Complex_Interp, construct_interpolations, gradient

export cpvagk

export inst_population, gibbs_state, eigen_sys, low_level_hamiltonian

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

export eigen_decomp

export AbstractAnnealing,
       Annealing,
       ODEParams,
       AbstractAnnealingControl

export AbstractBath,
       AbstractOpenSys,
       OpenSysSets,
       Redfield,
       RedfieldSet,
       DaviesGenerator,
       AMEDiffEqOperator,
       AMETrajectoryOperator,
       AFRWADiffEqOperator

export ame_jump

export UnitTime

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
