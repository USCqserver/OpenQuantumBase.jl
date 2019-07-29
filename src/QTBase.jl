module QTBase

using DocStringExtensions

import LinearAlgebra:kron, mul!, axpy!, I, ishermitian, Hermitian, eigmin, eigen, tr, eigen!, axpy!, diag, lmul!, Diagonal, normalize
import LinearAlgebra.BLAS:her!, gemm!
import SparseArrays:sparse, issparse, spzeros, SparseMatrixCSC
import Arpack:eigs
import QuadGK:quadgk
import Interpolations:interpolate, BSpline, Quadratic, Line, OnGrid, scale, gradient1, extrapolate


"""
$(TYPEDEF)

Suptertype for Hamiltonians with elements of type `T`. Any Hamiltonian object should implement two interfaces: `H(t)` and `H(du, u, p, t)`.
"""
abstract type AbstractHamiltonian{T <: Number} end


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
abstract type AbstractAnnealing{hType <: AbstractHamiltonian,uType <: Union{Vector,Matrix}} end


"""
$(TYPEDEF)

Base for types defining various control protocols in quantum annealing process.
"""
abstract type AbstractAnnealingControl end


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

include("integration/cpvagk.jl")

#include("controls.jl")
include("hamiltonian/affine_operator.jl")
include("hamiltonian/dense_hamiltonian.jl")
include("hamiltonian/sparse_hamiltonian.jl")
include("hamiltonian/adiabatic_frame_hamiltonian.jl")
include("hamiltonian/piecewise_hamiltonian.jl")
include("hamiltonian/util.jl")


include("opensys/redfield.jl")
include("opensys/davies.jl")


include("annealing/annealing_type.jl")
include("annealing/annealing_params.jl")
include("annealing/displays.jl")


export temperature_2_beta, temperature_2_freq, beta_2_temperature, freq_2_temperature

export σx, σz, σy, σi, σ, ⊗, PauliVec, spσx, spσz, spσi, spσy

export q_translate, construct_hamming_weight_op, single_clause, standard_driver, collective_operator, GHZ_entanglement_witness, local_field_term, two_local_term, q_translate_state

export matrix_decompose, check_positivity, check_unitary

export Complex_Interp, construct_interpolations

export cpvagk

export inst_population, gibbs_state, eigen_sys, low_level_hamiltonian

export AbstractHamiltonian, AbstractSparseHamiltonian, SparseHamiltonian, AbstractDenseHamiltonian, DenseHamiltonian, AdiabaticFrameHamiltonian, PiecewiseHamiltonian, evaluate

export eigen_decomp, p_copy

export Annealing, AnnealingParams

#export AdiabaticFramePiecewiseControl

export AbstractOpenSys, OpenSysSets, Redfield, Davies

end  # module QTBase
