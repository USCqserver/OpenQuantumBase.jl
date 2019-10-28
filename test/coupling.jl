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

c = ConstantCouplings(["ZI", "IZ"], sp=true)
@test c.mats[1] == 2π*spσz⊗spσi
@test c.mats[2] == 2π*spσi⊗spσz
