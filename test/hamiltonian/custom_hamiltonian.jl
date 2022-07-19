using OpenQuantumBase, Test
import LinearAlgebra: Diagonal

f(s) = (1 - s) * σx + s * σz
H = hamiltonian_from_function(f)

@test H(0.5) == 0.5 * (σx + σz)
@test !isconstant(H)

cache = get_cache(H)
update_cache!(cache, H, 2.0, 0.5)
@test cache == -0.5im * (σx + σz)

du = zeros(ComplexF64, 2, 2)
ρ = PauliVec[1][1] * PauliVec[1][1]'
H(du, ρ, nothing ,0.5)
@test du ≈ -1.0im * (f(0.5) * ρ - ρ * f(0.5))

w, v = haml_eigs(H, 0.5, 2)
@test v' * Diagonal(w) * v ≈ H(0.5)