module QTBase

import LinearAlgebra:kron, mul!, axpy!, I, ishermitian, Hermitian, eigmin, eigen, tr, eigen!, axpy!
import LinearAlgebra.BLAS:her!, gemm!
import SparseArrays:sparse, issparse, spzeros, SparseMatrixCSC
import Arpack:eigs

export temperature_2_beta, temperature_2_freq, beta_2_temperature, freq_2_temperature

export σx, σz, σy, σi, σ, ⊗, PauliVec, spσx, spσz, spσi, spσy

export q_translate, construct_hamming_weight_op, ising_terms, standard_driver, collective_operator, GHZ_entanglement_witness, local_field_term, two_local_term

export matrix_decompose, check_positivity, check_unitary

export inst_population, gibbs_state, eigen_eval, eigen_state_continuation!, low_level_hamiltonian, minimum_gap

export LinearOperator, update!, comm!, AdiabaticFrameHamiltonian
export set_tf!, construct_pausing_hamiltonian
export UnitlessAdiabaticFrameHamiltonian, UnitAdiabaticFrameHamiltonian, UnitlessAdiabaticFramePausingHamiltonian, UnitAdiabaticFramePausingHamiltonian

include("unit_util.jl")
include("math_util.jl")
include("matrix_util.jl")
include("linear_operator.jl")
include("hamiltonian.jl")

end  # module QTBase
