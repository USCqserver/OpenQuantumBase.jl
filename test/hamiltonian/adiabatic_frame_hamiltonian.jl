using QTBase, Test

funcs = [(x) -> x, (x) -> 1 - x]
test_diag_operator = QTBase.DiagonalOperator(funcs)
@test test_diag_operator(0.5) == QTBase.Diagonal([0.5, 0.5])

test_geometric_operator = QTBase.GeometricOperator(((x) -> -1.0im * x))
@test test_geometric_operator(0.5) == [0 0.5im; -0.5im 0]
@test_throws ArgumentError QTBase.GeometricOperator((x) -> x, (x) -> 1 - x)


dθ = (s) -> π / 2
gap = (s) -> (cos(2 * π * s) + 1) / 2
H = AdiabaticFrameHamiltonian([(x) -> -gap(x), (x) -> gap(x)], [dθ])
u = PauliVec[2][1]
ρ = u * u'

@test get_cache(H) == zeros(eltype(H), 2, 2)
@test size(H) == (2, 2)
@test H(10, 0.5) ≈ π * σx / 20
@test H(5, 0.0) ≈ π * σx / 10 - 2π * σz

# in_place update for vector
cache = get_cache(H)
update_cache!(cache, H, 10, 0.0)
@test cache ≈ -1.0im * (π * σx / 20 - 2π * σz)
update_cache!(cache, H, 10, 0.5)
@test cache ≈ -1.0im * (π * σx / 20)

# in_place update for matrices
du = [1.0 + 0.0im 0; 0 0]
hres = π * σx / 20 - 2π * σz
H(du, ρ, 10, 0.0)
@test du ≈ -1.0im * (hres * ρ - ρ * hres)
@test QTBase.ω_matrix(H, 2) ≈ [0 2; -2 0] * 2 * π
