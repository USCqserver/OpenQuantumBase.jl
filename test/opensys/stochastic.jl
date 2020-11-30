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
random_1 = rand([-1, 1], 2, 1)
τ, idx = findmin(rand(fluct.dist, 1))
Random.seed!(1234)
QTBase.reset!(fluct, (x, y) -> rand([-1, 1], x, y))

random_1[idx] *= -1
@test fluct.b0[1] == random_1[1] && fluct.b0[2] == random_1[2]
@test fluct.next_τ ≈ τ
@test fluct.next_idx == idx

b0 = copy(fluct.b0)
Random.seed!(1234)
random_1 = rand(fluct.dist, 1)
τ, idx = findmin(random_1)
b0[idx] *= -1
Random.seed!(1234)
QTBase.next_state!(fluct)
@test fluct.b0[1] == b0[1] && fluct.b0[2] == b0[2]
@test fluct.next_τ ≈ τ
@test fluct.next_idx == idx