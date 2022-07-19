"""
$(TYPEDEF)

Defines a time dependent dense Hamiltonian object with custom function.

# Fields

$(FIELDS)
"""
struct CustomDenseHamiltonian{T<:Number,dimensionless_time,in_place} <: AbstractHamiltonian{T}
    """Function for the Hamiltonian `H(s)`"""
    f::Any
    """Size"""
    size::Tuple
end


function hamiltonian_from_function(func; in_place=false, dimensionless_time=true)
    hmat = func(0.0)
    CustomDenseHamiltonian{eltype(hmat),dimensionless_time,in_place}(func, size(hmat))
end

get_cache(H::CustomDenseHamiltonian) = zeros(eltype(H), size(H))

(H::CustomDenseHamiltonian)(s) = H.f(s)

function update_cache!(cache, H::CustomDenseHamiltonian{T,dt,false}, ::Any, s::Real) where {T,dt}
    cache .= -1.0im * H(s)
end

function update_cache!(cache, H::CustomDenseHamiltonian{T,dt,true}, ::Any, s::Real) where {T,dt}
    H.f(cache, s)
end

function update_vectorized_cache!(cache, H::CustomDenseHamiltonian, ::Any, s::Real)
    hmat = H(s)
    iden = one(hmat)
    cache .= 1.0im * (transpose(hmat) ⊗ iden - iden ⊗ hmat)
end

function (h::CustomDenseHamiltonian{T,dt,false})(du, u::AbstractMatrix, ::Any, s::Real) where {T,dt}
    fill!(du, 0.0 + 0.0im)
    H = h(s)
    gemm!('N', 'N', -1.0im, H, u, 1.0 + 0.0im, du)
    gemm!('N', 'N', 1.0im, u, H, 1.0 + 0.0im, du)
end

function (h::CustomDenseHamiltonian{T,dt,true})(du, u::AbstractMatrix, p::Any, s::Real) where {T,dt}
    H.f(du, u, p, s)
end