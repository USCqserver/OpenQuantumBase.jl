using OpenQuantumBase, Test

η = 1e-4;T = 16
X = ConstantCouplings(["Z"])
bath_x = Ohmic(η, 1, T)
Z = ConstantCouplings(["X"])
bath_z = Ohmic(η, 4, T)

inter_x = Interaction(X, bath_x)
inter_z = Interaction(Z, bath_z)

inter_set = InteractionSet(inter_x, inter_z)

U(t) = exp(-1.0im * σz * t)
CGL = OpenQuantumBase.cg_from_interactions(inter_set, U, 10, 10, 1e-4, 1e-4)
@test length(CGL) == 1
@test CGL[1].kernels[1][2] == X
@test CGL[1].kernels[2][2] == Z
@test CGL[1].kernels[1][3][1,1](0.2, 0.1) ≈ correlation(0.1, bath_x)