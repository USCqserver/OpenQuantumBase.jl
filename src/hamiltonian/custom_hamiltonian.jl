"""
$(TYPEDEF)

Defines a time dependent Hamiltonian object with custom function.

# Fields

$(FIELDS)
"""
struct CustomDenseHamiltonian{T<:Number,in_place} <: AbstractDenseHamiltonian{T}
    """Function for the Hamiltonian `H(s)`"""
    f::Any
    """Internal cache"""
    u_cache::Any
    """Size"""
    size::Any
end


function hamiltonian_from_function(func; in_place = false)
    hmat = func(0.0)
    CustomDenseHamiltonian{eltype(hmat),in_place}(func, nothing, size(hmat))
end

get_cache(H::CustomDenseHamiltonian) = zeros(eltype(H), size(H))
(H::CustomDenseHamiltonian)(s) = H.f(s)
