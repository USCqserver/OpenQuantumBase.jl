using QTBase, Test

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
# === unitary test ===
u_res = exp(-1.0im*5*0.5*σx)
@test check_unitary(u_res)
@test !check_unitary([0 1; 0 0])

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
