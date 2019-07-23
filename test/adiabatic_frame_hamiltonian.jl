using QTBase, Test

dθ = (s)->π / 2
gap = (s)->(cos(2 * π * s) + 1) / 2
H = AdiabaticFrameHamiltonian([gap], [-σz], [dθ], [-σx])
u = [1.0 + 0.0im, 1] / sqrt(2)
ρ = u * u'

@test H(10, 0.5) ≈ -π^2 * σx
@test H(5, 0.0) ≈ -π^2 * σx - 10π * σz

du = [1.0 + 0.0im 0; 0 0]
hres = -π^2 * σx - 20π * σz
H(du, ρ, 10, 0.0)
@test du ≈ -1.0im * (hres * ρ - ρ * hres) + [1.0 + 0.0im 0; 0 0]
@test QTBase.ω_matrix(H) ≈ [0 2; -2 0] * 2 * π

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
