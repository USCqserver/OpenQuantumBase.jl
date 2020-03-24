using QTBase, Test

f(s) = (1 - s) * σx + s * σz
H = hamiltonian_from_function(f)

@test H(0.5) == 0.5 * (σx + σz)
@test H(2, 0.5) == (σx + σz)
@test H(UnitTime(10), 5) == 0.5 * (σx + σz)

cache = get_cache(H)
update_cache!(cache, H, 2.0, 0.5)
@test cache == -1.0im * (σx + σz)
update_cache!(cache, H, UnitTime(10), 5)
@test cache == -0.5im * (σx + σz)
