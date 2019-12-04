"""
$(TYPEDEF)

Defines interpolating DenseHamiltonian object

# Fields

$(FIELDS)
"""
struct InterpDenseHamiltonian{T} <: AbstractDenseHamiltonian{T}
    """Interpolating object"""
    interp_obj
    """Size"""
    size
end


"""
$(TYPEDEF)

Defines interpolating SparseHamiltonian object

# Fields

$(FIELDS)
"""
struct InterpSparseHamiltonian{T} <: AbstractSparseHamiltonian{T}
    """Interpolating object"""
    interp_obj
    """Size"""
    size
end


function InterpDenseHamiltonian(
    s,
    hmat;
    method = "bspline",
    order = 1,
    unit = :h,
)
    if ndims(hmat) == 3
        hsize = size(hmat)[1:2]
        htype = eltype(hmat)
    elseif ndims(hmat) == 1
        hsize = size(hmat[1])
        htype = eltype(sum(hmat))
    else
        throw(ArgumentError("Invalid input data dimension."))
    end

    interp_obj = construct_interpolations(
        s,
        unit_scale(unit) * hmat,
        method = method,
        order = order,
    )
    InterpDenseHamiltonian{htype}(interp_obj, hsize)
end


function (H::InterpDenseHamiltonian)(s)
    H.interp_obj(1:size(H, 1), 1:size(H, 1), s)
end


function (H::InterpDenseHamiltonian)(tf::Real, s)
    tf * H.interp_obj(1:size(H, 1), 1:size(H, 1), s)
end


function (H::InterpDenseHamiltonian)(tf::UnitTime, t)
    s = t / tf
    H.interp_obj(1:size(H, 1), 1:size(H, 1), s)
end


function update_cache!(cache, H::InterpDenseHamiltonian, tf::Real, s::Real)
    @inbounds for i = 1:size(H, 1)
        for j = 1:size(H, 1)
            cache[i, j] = -1.0im * tf * H.interp_obj(i, j, s)
        end
    end
end


function update_cache!(cache, H::InterpDenseHamiltonian, tf::UnitTime, t::Real)
    s = t / tf
    @inbounds for i = 1:size(H, 1)
        for j = 1:size(H, 1)
            cache[i, j] = -1.0im * H.interp_obj(i, j, s)
        end
    end
end


function update_vectorized_cache!(cache, H::InterpDenseHamiltonian, tf, t)
    hmat = H(tf, t)
    iden = Matrix{eltype(H)}(I, size(H))
    cache .= 1.0im * (transpose(hmat) ⊗ iden - iden ⊗ hmat)
end


function get_cache(H::InterpDenseHamiltonian{T}, vectorize) where {T}
    if vectorize == true
        hsize = size(H, 1) * size(H, 1)
        Matrix{T}(undef, hsize, hsize)
    else
        Matrix{T}(undef, size(H))
    end
end


function (h::InterpDenseHamiltonian)(
    du,
    u::Matrix{T},
    p::Real,
    t::Real,
) where {T<:Complex}
    fill!(du, 0.0 + 0.0im)
    H = h(t)
    gemm!('N', 'N', -1.0im * p, H, u, 1.0 + 0.0im, du)
    gemm!('N', 'N', 1.0im * p, u, H, 1.0 + 0.0im, du)
end


function (h::InterpDenseHamiltonian)(
    du,
    u::Matrix{T},
    tf::UnitTime,
    t::Real,
) where {T<:Complex}
    fill!(du, 0.0 + 0.0im)
    H = h(t / tf)
    gemm!('N', 'N', -1.0im, H, u, 1.0 + 0.0im, du)
    gemm!('N', 'N', 1.0im, u, H, 1.0 + 0.0im, du)
end


function InterpSparseHamiltonian(
    s_axis,
    H_list::AbstractArray{SparseMatrixCSC{T,Int},1};
    unit = :h,
) where {T<:Number}
    interp_obj = construct_interpolations(
        collect(s_axis),
        unit_scale(unit) * H_list,
        method = "gridded",
        order = 1,
    )
    InterpSparseHamiltonian{T}(interp_obj, size(H_list[1]))
end


function (H::InterpSparseHamiltonian)(s)
    H.interp_obj(s)
end


function (H::InterpSparseHamiltonian)(tf::Real, s)
    tf * H.interp_obj(s)
end


function (H::InterpSparseHamiltonian)(tf::UnitTime, t)
    s = t / tf
    H.interp_obj(s)
end


function get_cache(H::InterpSparseHamiltonian{T}, vectorize) where {T}
    if vectorize == true
        hsize = size(H, 1) * size(H, 1)
        spzeros(T, hsize, hsize)
    else
        spzeros(T, size(H)...)
    end
end


function update_cache!(cache, H::InterpSparseHamiltonian, tf, t::Real)
    cache .= -1.0im * H(tf, t)
end


function update_vectorized_cache!(
    cache,
    H::InterpSparseHamiltonian,
    tf,
    t::Real,
)
    hmat = H(tf, t)
    iden = sparse(I, size(H))
    cache .= 1.0im * (transpose(hmat) ⊗ iden - iden ⊗ hmat)
end


function (h::InterpSparseHamiltonian)(
    du,
    u::Matrix{T},
    tf::Real,
    t::Real,
) where {T<:Number}
    H = h(t)
    du .= -1.0im * tf * (H * u - u * H)
end


function (h::InterpSparseHamiltonian)(
    du,
    u::Matrix{T},
    tf::UnitTime,
    t::Real,
) where {T<:Number}
    H = h(t / tf)
    du .= -1.0im * (H * u - u * H)
end
