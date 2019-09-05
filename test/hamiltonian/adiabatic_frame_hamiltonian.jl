using QTBase, Test

funcs = [(x)->x, (x)->1-x]
test_diag_operator = QTBase.DiagonalOperator(funcs)
@test test_diag_operator(0.5) == QTBase.Diagonal([0.5, 0.5])

test_geometric_operator = QTBase.GeometricOperator(((x)->-1.0im*x))
@test test_geometric_operator(0.5) == [0 0.5im; -0.5im 0]
@test_throws ArgumentError QTBase.GeometricOperator((x)->x, (x)->1-x)


dθ = (s)->π / 2
gap = (s)->(cos(2 * π * s) + 1) / 2
H = AdiabaticFrameHamiltonian([(x)->-gap(x), (x)->gap(x)], [dθ])
u = PauliVec[2][1]
ρ = u * u'

@test H(UnitTime(10), 5.0) ≈ π * σx / 20
@test H(UnitTime(5), 0.0) ≈ π * σx / 10 - 2π * σz

@test H(10, 0.5) ≈ π * σx / 2
@test H(5, 0.0) ≈ π * σx / 2 - 10π * σz

# in_place update for vector
du = [1.0 + 0.0im, 0]
H(du, u, 10, 0.0)
@test du ≈ -1.0im*(π * σx / 2 - 20π * σz) * u
H(du, u, UnitTime(10), 0.0)
@test du ≈ -1.0im*(π * σx / 20 - 2π * σz) * u
H(du, u, UnitTime(10), 5)
@test du ≈ -1.0im*(π * σx / 20) * u

# in_place update for matrices
du = [1.0 + 0.0im 0; 0 0]
hres = π * σx / 2 - 20π * σz
H(du, ρ, 10, 0.0)
@test du ≈ -1.0im * (hres * ρ - ρ * hres)
@test QTBase.ω_matrix(H, 2) ≈ [0 2; -2 0] * 2 * π

#data_ρ = QTBase.StateMachineDensityMatrix(ρ, 1)
#control = QTBase.AdiabaticFramePauseControl(10, [0.5], [(x)->x, (x)->0.0], [1, 0.0])

# du = [1.0+0.0im 0; 0 0]
# data_du = QTBase.StateMachineDensityMatrix(du, 1)
# H(data_du, data_ρ, control, 0.0)
# @test data_du ≈ -1.0im*(hres*ρ-ρ*hres) + [1.0+0.0im 0; 0 0]
#
# data_ρ.state += 1
# du = [1.0+0.0im 0; 0 0]
# data_du = QTBase.StateMachineDensityMatrix(du, 1)
# hres = -10σz
# H(data_du, data_ρ, control, 0.5)
# @test data_du ≈ -1.0im*(hres*ρ-ρ*hres) + [1.0+0.0im 0; 0 0]
