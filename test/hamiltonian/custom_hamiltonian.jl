using OpenQuantumBase, Test

f(s) = (1 - s) * σx + s * σz
H = hamiltonian_from_function(f)

@test H(0.5) == 0.5 * (σx + σz)
@test !isconstant(H)

cache = get_cache(H)
update_cache!(cache, H, 2.0, 0.5)
@test cache == -0.5im * (σx + σz)
