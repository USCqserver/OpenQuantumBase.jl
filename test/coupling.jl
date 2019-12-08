using QTBase, Test

c = ConstantCouplings(["ZI", "IZ"])
@test c.mats[1] == 2π*σz⊗σi
@test c.mats[2] == 2π*σi⊗σz
res = c(0.2)
@test res[1] == 2π*σz⊗σi
@test res[2] == 2π*σi⊗σz

c = ConstantCouplings(["ZI", "IZ"], unit=:ħ)
@test c.mats[1] == σz⊗σi
@test c.mats[2] == σi⊗σz
res = c(0.2)
@test res[1] == σz⊗σi
@test res[2] == σi⊗σz
@test [op for op in c] == [σz⊗σi, σi⊗σz]

c = ConstantCouplings(["ZI", "IZ"], sp=true)
@test c.mats[1] == 2π*spσz⊗spσi
@test c.mats[2] == 2π*spσi⊗spσz

c1 = TimeDependentCoupling([(s)->s], [σz], unit=:ħ)
@test c1(0.5) == 0.5σz
c2 = TimeDependentCoupling([(s)->s], [σx], unit=:ħ)
c = TimeDependentCouplings(c1, c2)
@test [op for op in c(0.5)] == [c1(0.5), c2(0.5)]

c = collective_coupling("Z", 2, unit=:ħ)
@test c(0.1) == [σz⊗σi, σi⊗σz]
