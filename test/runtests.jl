using QTBase, Test

@testset "QTBase" begin

# === matrix decomposition
v = 1.0*σx + 2.0*σy + 3.0*σz
res = matrix_decompose(v, [σx,σy,σz])
@test isapprox(res, [1.0,2.0,3.0])
# === positivity test ===
r = rand(2)
m = r[1]*PauliVec[1][2]*PauliVec[1][2]' + r[2]*PauliVec[1][1]*PauliVec[1][1]'
@test check_positivity(m)
@test !check_positivity(σx)
# == units conversion test ===
@test isapprox(temperature_2_freq(1e3), 20.8366176361328, atol=1e-4, rtol=1e-4)
# == Hamiltonian analysis ===
hfun(s) = (1-s)*real(σx)+ s*real(σz)
dhfun(s) = -real(σx) + real(σz)
t = [0.0, 1.0]
states = [PauliVec[1][2], PauliVec[1][1]]
res = inst_population(t, states, hfun, level=1:2)
@test isapprox(res, [[1.0,0],[0.5,0.5]])
hfun(s) = -(1-s)*standard_driver(2) + s * (0.1*σz⊗σi + σz⊗σz)
sphfun(s) = real(-(1-s)*standard_driver(2,sp=true) + s * (0.1*spσz⊗spσi + spσz⊗spσz))
spdhfun(s) = real(standard_driver(2,sp=true) + (0.1*spσz⊗spσi + spσz⊗spσz))
interaction = [spσz⊗spσi, spσi⊗spσz]
spw, spv = eigen_eval(sphfun, [0.5])
w, v = eigen_eval(hfun, [0.5], levels=2)
@test isapprox(w, spw, atol=1e-4)
@test isapprox(spv[:,1,1], v[:,1,1], atol=1e-4) || isapprox(spv[:,1,1], -v[:,1,1], atol=1e-4)
@test isapprox(spv[:,2,1], v[:,2,1], atol=1e-4) || isapprox(spv[:,2,1], -v[:,2,1], atol=1e-4)
# === unitary test ===
u_res = exp(-1.0im*5*0.5*σx)
@test check_unitary(u_res)
@test !check_unitary([0 1; 0 0])

end

@testset "Ising Hamiltonian Construction" begin

@test ising_terms(["x"],[2],0.5,2) == 0.5*σi⊗σx
@test ising_terms(["z","z"],[2,3],-2,4) == -2*σi⊗σz⊗σz⊗σi
@test standard_driver(2) == σx⊗σi + σi⊗σx
@test collective_operator("z", 3) ≈ σz⊗σi⊗σi + σi⊗σz⊗σi + σi⊗σi⊗σz
@test local_field_term([-1.0, 0.5], [1,3], 3) ≈ -1.0*σz⊗σi⊗σi + 0.5*σi⊗σi⊗σz
@test local_field_term([-1.0, 0.5], [1,3], 3, sp=true) ≈ -1.0*spσz⊗spσi⊗spσi + 0.5*spσi⊗spσi⊗spσz
@test two_local_term([-1.0, 0.5], [[1,3],[1,2]], 3) ≈ -1.0*σz⊗σi⊗σz + 0.5*σz⊗σz⊗σi

end

# @testset "LinearOp" begin

# pausing_test = construct_pausing_hamiltonian(0.5, 1.0, h_test)
# @test pausing_test.tf_ext == 20
# @test pausing_test(0.6) ≈ zeros(2, 2)
# @test pausing_test(1.6) ≈ -π*real(σx)/2-(cos(1.2*π)+1)*10*real(σz)
# h_test = AdiabaticFrameHamiltonian([dθ], [gap], [-real(σx)], [-real(σz)], unitless=false)
# set_tf!(h_test, 10)
# @test h_test(5) ≈ -π * real(σx) / 20
# pausing_test = construct_pausing_hamiltonian(0.5, 1.0, h_test)
# @test pausing_test(5) ≈ -π * real(σx)/40
# @test pausing_test(6) ≈ zeros(2, 2)
# @test pausing_test(16) ≈ -π*real(σx)/40-(cos(1.2*π)+1)*real(σz)/2
#
# end


@testset "Adiabatic Frame Hamiltonian" begin

dθ= (s)->π/2
gap = (s)-> (cos(2*π*s) + 1)/2
H = hamiltonian_factory([gap], [-σz], [dθ], [-σx])
u = [1.0+0.0im, 1]/sqrt(2)
ρ = u*u'
data_ρ = ControlDensityMatrix(ρ, 1)
control = AdiabaticFramePiecewiseControl(10, 1.0, [0.5], [(x)->x, (x)->0.0], [1, 0.0])

@test H(10, 0.5) ≈ -π*σx/2
@test H(5, 0.0) ≈ -π*σx/2 - 5σz

du = [1.0+0.0im 0; 0 0]
hres = -π*σx/2 - 10σz
@test H(du, ρ, 10, 0.0) ≈ -1.0im*(hres*ρ-ρ*hres) + [1.0+0.0im 0; 0 0]
@test QTBase.ω_matrix(H) ≈ [0 2; -2 0]

#du = [1.0+0.0im 0; 0 0]
#@test H(du, ρ, control, 0.0) ≈ -1.0im*(hres*ρ-ρ*hres) + [1.0+0.0im 0; 0 0]

du = [1.0+0.0im 0; 0 0]
data_du = ControlDensityMatrix(du, 1)
@test H(data_du, data_ρ, control, 0.0) ≈ -1.0im*(hres*ρ-ρ*hres) + [1.0+0.0im 0; 0 0]

data_ρ.stage += 1
du = [1.0+0.0im 0; 0 0]
data_du = ControlDensityMatrix(du, 1)
hres = -10σz
@test H(data_du, data_ρ, control, 0.5) ≈ -1.0im*(hres*ρ-ρ*hres) + [1.0+0.0im 0; 0 0]

scale!(H, 10, "adiabatic")
@test H(1.0, 0.0) ≈ -π*σx/2 - 10σz
scale!(H, 10, "geometric")
@test H(1.0, 0.0) ≈ -5π*σx - 10σz

end

@testset "Time Dependent Hamiltonian" begin

A = (s)->(1-s)
B = (s)->s
u = [1.0+0.0im, 1]/sqrt(2)
ρ = u*u'

H = hamiltonian_factory([A, B], [σx, σz])
H_sparse = hamiltonian_factory([A, B], [spσx, spσz])

@test H(0) ≈ σx
@test H_sparse(0) ≈ spσx
@test H(0.5) ≈ 0.5σx + 0.5σz
@test H_sparse(0.5) ≈ 0.5spσx + 0.5spσz

du = [1.0+0.0im 0; 0 0]
H(du, ρ, 0.5)
@test du ≈ -1.0im*((0.5σx + 0.5σz)*ρ -ρ*(0.5σx + 0.5σz)) + [1.0+0.0im 0; 0 0]

du = [1.0+0.0im 0; 0 0]
H_sparse(du, ρ, 0.5)
@test du ≈ -1.0im*((0.5σx + 0.5σz)*ρ -ρ*(0.5σx + 0.5σz)) + [1.0+0.0im 0; 0 0]

scale!(H, 2)
@test H(0.5) ≈ σx + σz
scale!(H_sparse, 2)
@test H_sparse(0.5) ≈ spσx + spσz

end

@testset "Adiabatic Frame Pausing Annealing" begin
dθ= (s)->π/2
gap = (s)-> 1.0
H = hamiltonian_factory([gap], [-σz], [dθ], [-σx])
u0 = [1.0+0.0im, 0]
normal_anneal = annealing_factory(H, u0)
piecewise_anneal = annealing_factory(H, u0, 0.5, 1.0)

@test piecewise_anneal.ode_problem.p.stops == [0.5, 1.5]
end
