using QTBase, Test

@test q_translate("ZZ+0.5ZI-XZ") == σz⊗σz + 0.5σz⊗σi - σx⊗σz
@test single_clause(["x"], [2], 0.5, 2) == 0.5σi⊗σx
@test single_clause(["x"], [2], 0.5, 2, sp = true) == 0.5spσi⊗spσx
@test single_clause(["z","z"], [2,3], -2, 4) == -2σi⊗σz⊗σz⊗σi
@test standard_driver(2) == σx⊗σi + σi⊗σx
@test standard_driver(2, sp = true) == spσx⊗spσi + spσi⊗spσx
@test collective_operator("z", 3) ≈ σz⊗σi⊗σi + σi⊗σz⊗σi + σi⊗σi⊗σz
@test collective_operator("z", 3, sp = true) ≈ spσz⊗spσi⊗spσi + spσi⊗spσz⊗spσi + spσi⊗spσi⊗spσz
@test local_field_term([-1.0, 0.5], [1,3], 3) ≈ -1.0σz⊗σi⊗σi + 0.5σi⊗σi⊗σz
@test local_field_term([-1.0, 0.5], [1,3], 3, sp = true) ≈ -1.0spσz⊗spσi⊗spσi + 0.5spσi⊗spσi⊗spσz
@test two_local_term([-1.0, 0.5], [[1,3],[1,2]], 3) ≈ -1.0σz⊗σi⊗σz + 0.5σz⊗σz⊗σi

v0 = [1.0+0.0im, 0]
v1 = [0, 1.0+0.0im]
@test q_translate_state("0011") == v0⊗v0⊗v1⊗v1
@test q_translate_state("0.1(111)+(011)") == 0.1*v1⊗v1⊗v1 + v0⊗v1⊗v1
@test q_translate_state("(111)+(011)"; normal=true) == (v1⊗v1⊗v1 + v0⊗v1⊗v1)/sqrt(2)
