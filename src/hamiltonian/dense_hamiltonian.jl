"""
$(TYPEDEF)

Defines a time dependent Hamiltonian object with dense Matrices.

# Fields

$(FIELDS)
"""
struct DenseHamiltonian{T <: Number} <: AbstractDenseHamiltonian{T}
    " List of time dependent functions "
    f
    " List of constant matrices "
    m::Vector{Matrix{T}}
    """Internal cache"""
    u_cache::Matrix{T}
    """Size"""
    size
end


"""
    function DenseHamiltonian(funcs, mats; unit = :h)

Constructor of DenseHamiltonian object. `funcs` and `mats` are a list of time dependent functions and the corresponding matrices. `unit` is the unit one -- `:h` or `:ħ`. The `mats` will be scaled by ``2π`` if unit is `:h`.
"""
function DenseHamiltonian(funcs, mats; unit = :h)
    if !all((x)->size(x) == size(mats[1]), mats)
        throw(ArgumentError("Matrices in the list do not have the same size."))
    end
    cache = zeros(eltype(mats[1]), size(mats[1]))
    DenseHamiltonian(funcs, unit_scale(unit)*mats, cache, size(mats[1]))
end


"""
    function (h::DenseHamiltonian)(t::Real)

Calling the Hamiltonian returns the value ``2πH(t)``.
"""
function (h::DenseHamiltonian)(t::Real)
    fill!(h.u_cache, 0.0)
    for (f, m) in zip(h.f, h.m)
        axpy!(f(t), m, h.u_cache)
    end
    h.u_cache
end


function (h::DenseHamiltonian)(tf::Real, t::Real)
    hmat = h(t)
    lmul!(tf, hmat)
end


function (h::DenseHamiltonian)(tf::UnitTime, t::Real)
    hmat = h(t/tf)
end


function (h::DenseHamiltonian)(du, u::Matrix{T}, p::Real, t::Real) where T<:Complex
    fill!(du, 0.0+0.0im)
    H = h(t)
    gemm!('N', 'N', -1.0im * p, H, u, 1.0 + 0.0im, du)
    gemm!('N', 'N', 1.0im * p, u, H, 1.0 + 0.0im, du)
end


function (h::DenseHamiltonian)(du, u::Matrix{T}, tf::UnitTime, t::Real) where T<:Complex
    fill!(du, 0.0+0.0im)
    H = h(t/tf)
    gemm!('N', 'N', -1.0im, H, u, 1.0 + 0.0im, du)
    gemm!('N', 'N', 1.0im, u, H, 1.0 + 0.0im, du)
end


function (h::DenseHamiltonian)(du, u::Vector{T}, tf::Real, t::Real) where T<:Complex
    H = h(t)
    mul!(du, H, u)
    lmul!(-1.0im * tf, du)
end


function (h::DenseHamiltonian)(du, u::Vector{T}, tf::UnitTime, t::Real) where T<:Complex
    H = h(t/tf)
    mul!(du, H, u)
    lmul!(-1.0im, du)
end


"""
    function eigen_decomp(h::AbstractDenseHamiltonian, t; level = 2) -> (w, v)

Calculate the eigen value decomposition of the Hamiltonian `h` at time `t`. Keyword argument `level` specifies the number of levels to keep in the output. `w` is a vector of eigenvalues and `v` is a matrix of the eigenvectors in the columns. (The `k`th eigenvector can be obtained from the slice `v[:, k]`.) `w` will be in unit of `GHz`.
"""
function eigen_decomp(h::AbstractDenseHamiltonian, t; level = 2, kwargs...)
    H = h(t)
    w, v = eigen!(Hermitian(H), 1:level)
    lmul!(1 / 2 / π, w), v
end


function ode_eigen_decomp(h::AbstractDenseHamiltonian, lvl::Integer)
    eigen!(Hermitian(h.u_cache), 1:lvl)
end


function p_copy(h::DenseHamiltonian)
    DenseHamiltonian(h.f, h.m, similar(h.u_cache), h.size)
end
