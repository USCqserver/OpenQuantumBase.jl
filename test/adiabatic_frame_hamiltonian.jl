using QTBase, Test

dθ= (s)->π/2
gap = (s)-> (cos(2*π*s) + 1)/2
H = hamiltonian_factory([gap], [-σz], [dθ], [-σx])
u = [1.0+0.0im, 1]/sqrt(2)
ρ = u*u'
data_ρ = QTBase.ControlDensityMatrix(ρ, 1)
control = AdiabaticFramePiecewiseControl(10, 1.0, [0.5], [(x)->x, (x)->0.0], [1, 0.0])

@test H(10, 0.5) ≈ -π*σx/2
@test H(5, 0.0) ≈ -π*σx/2 - 5σz

du = [1.0+0.0im 0; 0 0]
hres = -π*σx/2 - 10σz
@test H(du, ρ, 10, 0.0) ≈ -1.0im*(hres*ρ-ρ*hres) + [1.0+0.0im 0; 0 0]
@test QTBase.ω_matrix(H) ≈ [0 2; -2 0]

du = [1.0+0.0im 0; 0 0]
data_du = QTBase.ControlDensityMatrix(du, 1)
@test H(data_du, data_ρ, control, 0.0) ≈ -1.0im*(hres*ρ-ρ*hres) + [1.0+0.0im 0; 0 0]

data_ρ.stage += 1
du = [1.0+0.0im 0; 0 0]
data_du = QTBase.ControlDensityMatrix(du, 1)
hres = -10σz
@test H(data_du, data_ρ, control, 0.5) ≈ -1.0im*(hres*ρ-ρ*hres) + [1.0+0.0im 0; 0 0]

QTBase.scale!(H, 10, "adiabatic")
@test H(1.0, 0.0) ≈ -π*σx/2 - 10σz
QTBase.scale!(H, 10, "geometric")
@test H(1.0, 0.0) ≈ -5π*σx - 10σz
