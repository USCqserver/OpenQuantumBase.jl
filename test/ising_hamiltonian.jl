using QTBase, Test

@test ising_terms(["x"],[2],0.5,2) == 0.5*σi⊗σx
@test ising_terms(["z","z"],[2,3],-2,4) == -2*σi⊗σz⊗σz⊗σi
@test standard_driver(2) == σx⊗σi + σi⊗σx
@test collective_operator("z", 3) ≈ σz⊗σi⊗σi + σi⊗σz⊗σi + σi⊗σi⊗σz
@test local_field_term([-1.0, 0.5], [1,3], 3) ≈ -1.0*σz⊗σi⊗σi + 0.5*σi⊗σi⊗σz
@test local_field_term([-1.0, 0.5], [1,3], 3, sp=true) ≈ -1.0*spσz⊗spσi⊗spσi + 0.5*spσi⊗spσi⊗spσz
@test two_local_term([-1.0, 0.5], [[1,3],[1,2]], 3) ≈ -1.0*σz⊗σi⊗σz + 0.5*σz⊗σz⊗σi
