module QTBase

using DocStringExtensions

import LinearAlgebra:kron, mul!, axpy!, I, ishermitian, Hermitian, eigmin, eigen, tr, eigen!, axpy!, diag, lmul!
import LinearAlgebra.BLAS:her!, gemm!
import SparseArrays:sparse, issparse, spzeros, SparseMatrixCSC
import Arpack:eigs
import DiffEqBase:DEDataVector, ODEProblem, ODEFunction,  DEDataMatrix, DiscreteCallback, u_modified!, full_cache, solve
import QuadGK:quadgk
import Interpolations:interpolate, BSpline, Quadratic, Line, OnGrid, scale, gradient1, extrapolate

"""
$(TYPEDEF)

Base for types defining Hamiltonians.
"""
abstract type AbstractHamiltonian{T<:Number} end

"""
$(TYPEDEF)

Base for types defining quantum annealing process.
"""
abstract type AbstractAnnealing end
abstract type AnnealingControl end
abstract type LinearOperator{T<:Number} end
abstract type LinearOperatorSparse{T<:Number} end
abstract type AbstractHamiltonian{T<:Number} end

include("unit_util.jl")
include("math_util.jl")
include("matrix_util.jl")

include("interpolation/interpolation.jl")
include("integration/integration.jl")
include("hamiltonian/hamiltonian.jl")
include("annealing/annealing.jl")

export temperature_2_beta, temperature_2_freq, beta_2_temperature, freq_2_temperature

export σx, σz, σy, σi, σ, ⊗, PauliVec, spσx, spσz, spσi, spσy

export q_translate, construct_hamming_weight_op, single_clause, standard_driver, collective_operator, GHZ_entanglement_witness, local_field_term, two_local_term

export matrix_decompose, check_positivity, check_unitary

export cpvagk

export inst_population, gibbs_state, eigen_eval, eigen_state_continuation!, low_level_hamiltonian, minimum_gap

export hamiltonian_factory, AbstractHamiltonian, Hamiltonian, HamiltonianSparse, AdiabaticFrameHamiltonian, eigen_decomp, calculate_unitary, solve_schrodinger, p_copy

export AdiabaticFramePiecewiseControl, annealing_factory

export update_tf!

export Complex_Interp, construct_interpolations


end  # module QTBase
