"""
$(TYPEDEF)

Defines a time dependent Hamiltonian object with custom function.

# Fields

$(FIELDS)
"""
struct CustomDenseHamiltonian{T<:Number,in_place} <: AbstractDenseHamiltonian{T}
    """Function for the Hamiltonian `H(s)`"""
    f
    """Internal cache"""
    u_cache
    """Size"""
    size
end


function hamiltonian_from_function(func; in_place = false)
    hmat = func(0.0)
    CustomDenseHamiltonian{eltype(hmat),in_place}(func, nothing, size(hmat))
end

get_cache(H::CustomDenseHamiltonian) = zeros(eltype(H), size(H))
(H::CustomDenseHamiltonian)(s) = H.f(s)
(H::CustomDenseHamiltonian)(tf::Real, s) = tf * H.f(s)
(H::CustomDenseHamiltonian)(tf::UnitTime, t) = H.f(t / tf)

update_cache!(cache, H::CustomDenseHamiltonian, tf, s) =
    cache .= -1.0im * H(tf, s)
