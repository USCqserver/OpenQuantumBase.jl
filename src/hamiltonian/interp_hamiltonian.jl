"""
$(TYPEDEF)

Defines interpolating DenseHamiltonian object.

# Fields

$(FIELDS)
"""
struct InterpDenseHamiltonian{T,isdimensionlesstime} <: AbstractDenseHamiltonian{T}
    "Interpolating object"
    interp_obj::Any
    "Size"
    size::Any
end


"""
$(TYPEDEF)

Defines interpolating SparseHamiltonian object.

# Fields

$(FIELDS)
"""
struct InterpSparseHamiltonian{T,dimensionless_time} <: AbstractSparseHamiltonian{T}
    "Interpolating object"
    interp_obj::Any
    "Size"
    size::Any
end

isdimensionlesstime(H::InterpDenseHamiltonian{T,true}) where {T} = true
isdimensionlesstime(H::InterpDenseHamiltonian{T,false}) where {T} = false
isdimensionlesstime(H::InterpSparseHamiltonian{T,true}) where {T} = true
isdimensionlesstime(H::InterpSparseHamiltonian{T,false}) where {T} = false

function InterpDenseHamiltonian(
    s,
    hmat;
    method="bspline",
    order=1,
    unit=:h,
    dimensionless_time=true
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
        method=method,
        order=order,
    )
    InterpDenseHamiltonian{htype,dimensionless_time}(interp_obj, hsize)
end

function (H::InterpDenseHamiltonian)(s)
    H.interp_obj(1:size(H, 1), 1:size(H, 1), s)
end

# The argument `p` is not essential for `InterpDenseHamiltonian`
# It exists to keep the `update_cache!` interface consistent across
# all `AbstractHamiltonian` types
function update_cache!(cache, H::InterpDenseHamiltonian, tf, s::Real)
    for i = 1:size(H, 1)
        for j = 1:size(H, 1)
            @inbounds cache[i, j] = -1.0im * H.interp_obj(i, j, s)
        end
    end
end

function update_vectorized_cache!(cache, H::InterpDenseHamiltonian, tf, s)
    hmat = H(s)
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

get_cache(H::InterpDenseHamiltonian{T}) where {T} = Matrix{T}(undef, size(H))


function (h::InterpDenseHamiltonian)(
    du,
    u::Matrix{T},
    ::Any,
    t::Real,
) where {T<:Complex}
    fill!(du, 0.0 + 0.0im)
    H = h(t)
    gemm!('N', 'N', -1.0im, H, u, 1.0 + 0.0im, du)
    gemm!('N', 'N', 1.0im, u, H, 1.0 + 0.0im, du)
end

function InterpSparseHamiltonian(
    s_axis,
    H_list::AbstractArray{SparseMatrixCSC{T,Int},1};
    unit=:h,
    dimensionless_time=true
) where {T<:Number}
    interp_obj = construct_interpolations(
        collect(s_axis),
        unit_scale(unit) * H_list,
        method="gridded",
        order=1,
    )
    InterpSparseHamiltonian{T, dimensionless_time}(interp_obj, size(H_list[1]))
end


function (H::InterpSparseHamiltonian)(s)
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

get_cache(H::InterpSparseHamiltonian{T}) where {T} = spzeros(T, size(H)...)

# The argument `p` is not essential for `InterpSparseHamiltonian`
# It exists to keep the `update_cache!` interface consistent across
# all `AbstractHamiltonian` types
update_cache!(cache, H::InterpSparseHamiltonian, p, s::Real) =
    cache .= -1.0im * H(s)

function update_vectorized_cache!(
    cache,
    H::InterpSparseHamiltonian,
    tf,
    s::Real,
)
    hmat = H(s)
    iden = sparse(I, size(H))
    cache .= 1.0im * (transpose(hmat) ⊗ iden - iden ⊗ hmat)
end

function (h::InterpSparseHamiltonian)(
    du,
    u::Matrix{T},
    tf,
    s::Real,
) where {T<:Number}
    H = h(s)
    du .= -1.0im * (H * u - u * H)
end
