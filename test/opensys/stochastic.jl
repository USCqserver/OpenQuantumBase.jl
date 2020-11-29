using QTBase, Random

bath = EnsembleFluctuator([1, 1], [1, 2])
interaction = InteractionSet(Interaction(ConstantCouplings(["Z"], unit=:ħ), bath))

fluct = QTBase.fluctuator_from_interactions(interaction)[1]
cache_exp = -1.0im * sum(fluct.b0 .* [σz, σz])

cache = zeros(ComplexF64, 2, 2)
p = ODEParams(nothing, 5.0, (tf, t) -> t / tf)
update_cache!(cache, fluct, p, 2.5)
@test cache ≈ cache_exp

cache = zeros(ComplexF64, 4, 4)
update_vectorized_cache!(cache, fluct, p, 2.5)
@test cache ≈ one(cache_exp)⊗cache_exp - transpose(cache_exp)⊗one(cache_exp)

Random.seed!(1234)
QTBase.reset!(fluct, (x, y) -> rand([-1, 1], x, y))
@test fluct.b0[1] == -1 && fluct.b0[2] == 1
@test fluct.next_τ ≈ 0.34543049081993993
@test fluct.next_idx == CartesianIndex(2, 1)

QTBase.next_state!(fluct)
@test fluct.b0[1] == 1 && fluct.b0[2] == 1
@test fluct.next_τ ≈ 1.3249941403673653
@test fluct.next_idx == CartesianIndex(1, 1)