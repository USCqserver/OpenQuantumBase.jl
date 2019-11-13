using QTBase, Test


H = DenseHamiltonian([(s) -> 1 - s, (s) -> s], [σx, σz])
dH(s) = (-real(σx) + real(σz))
coupling = ConstantCouplings(["Z"])

t_obj = project_to_lowlevel(H, dH, coupling, [0.0, 0.5, 1.0])

@test t_obj.ev ≈ [[-1.0, 1.0], [-0.707107, 0.707107], [-1.0, 1.0]] atol = 1e-6
@test t_obj.dθ ≈ [[0.5], [1.0], [0.5]]
@test get_dθ(t_obj) ≈ -[0.5, 1.0, 0.5]

@test t_obj.op ./ 2 ./ π ≈ [
    [[0 -1.0; -1.0 0]],
    [[-0.707107 -0.707107; -0.707107 0.707107]],
    [[-1.0 0.0; 0.0 1.0]],
] atol = 1e-6

    H = SparseHamiltonian(
    [(s) -> 1 - s, (s) -> s],
    [-standard_driver(2, sp = true) / 2, (spσz ⊗ spσi - 0.1spσz ⊗ spσz) / 2],
)
dH(s) = standard_driver(2, sp = true) / 2 + (spσz ⊗ spσi - 0.1spσz ⊗ spσz) / 2
coupling = ConstantCouplings(["ZI", "IZ"])

t_obj = project_to_lowlevel(H, dH, coupling, [0.0, 0.5, 1.0])

@test t_obj.ev ≈ [
    [-1.0, 0.0],
    [-0.6044361719689455, -0.10443617196894575],
    [-0.55, -0.45],
] atol = 1e-6

symmetric_matrix = QTBase.LinearIdxLowerTriangular(ComplexF64, 10, 2)
symmetric_matrix[:, 1, 2] = 1:10
@test symmetric_matrix[:, 2, 1] == 1:10
