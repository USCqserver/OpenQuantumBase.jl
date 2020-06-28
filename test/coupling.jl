using QTBase, Test

c = ConstantCouplings(["ZI", "IZ"])
@test isequal(c.mats[1], 2π*σz⊗σi)
@test c.mats[2](0) == 2π*σi⊗σz
res = c(0.2)
@test isequal(res[1], 2π*σz⊗σi)
@test res[2](0) == 2π*σi⊗σz

c = ConstantCouplings(["ZI", "IZ"], unit=:ħ)
@test isequal(c.mats[1], σz⊗σi)
@test isequal(c.mats[2], σi⊗σz)
res = c(0.2)
@test res[1](0) == σz⊗σi
@test res[2](0) == σi⊗σz
@test [op(0) for op in c] == [σz⊗σi, σi⊗σz]

c = ConstantCouplings(["ZI", "IZ"], sp=true)
@test isequal(c.mats[1], 2π*spσz⊗spσi)
@test isequal(c.mats[2], 2π*spσi⊗spσz)

c1 = TimeDependentCoupling([(s)->s], [σz], unit=:ħ)
@test c1(0.5) == 0.5σz
c2 = TimeDependentCoupling([(s)->s], [σx], unit=:ħ)
c = TimeDependentCouplings(c1, c2)
@test size(c) == (2, 2)
@test [op for op in c(0.5)] == [c1(0.5), c2(0.5)]

c = collective_coupling("Z", 2, unit=:ħ)
@test isequal(c(0.1), [σz⊗σi, σi⊗σz])


test_coupling = [(s)->s*σx, (s)->(1-s)*σz]
coupling = CustomCouplings(test_coupling, unit=:ħ)
@test size(coupling) == (2, 2)
@test coupling(0.5) == 0.5 * [σx, σz]
@test [c(0.2) for c in coupling] == [0.2*σx, 0.8*σz]
