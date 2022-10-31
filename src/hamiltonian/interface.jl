function ConstantHamiltonian(mat::Matrix; unit=:h, static=true)
    # use static array for size smaller than 100
    # can be turned off by setting `static` to false
    if static && size(mat, 1) <= 10
        mat = SMatrix{size(mat, 1),size(mat, 2)}(mat)
        return ConstantStaticDenseHamiltonian(mat, unit=unit)
    end
    ConstantDenseHamiltonian(mat, unit=unit)
end

ConstantHamiltonian(mat::Union{SMatrix,MMatrix}; unit=:h, static=true) = ConstantStaticDenseHamiltonian(mat, unit=unit)

function ConstantHamiltonian(mat::SparseMatrixCSC; unit=:h, static=true)
    mat = unit_scale(unit) * mat
    ConstantSparseHamiltonian(mat, size(mat))
end

function Hamiltonian(f, mats::AbstractVector{T}; unit=:h, dimensionless_time=true, static=true) where {T<:Matrix}
    hsize = size(mats[1])
    # use static array for size smaller than 100
    # can be turned off by setting `static` to false
    if static && hsize[1] <= 10
        mats = [SMatrix{hsize[1],hsize[2]}(unit_scale(unit) * m) for m in mats]
        return StaticDenseHamiltonian(f, mats, unit=unit, dimensionless_time=dimensionless_time)
    end
    DenseHamiltonian(f, mats, unit=unit, dimensionless_time=dimensionless_time)
end

Hamiltonian(f, mats::AbstractVector{T}; unit=:h, dimensionless_time=true, static=true) where {T<:Union{SMatrix,MMatrix}} = StaticDenseHamiltonian(f, mats, unit=unit, dimensionless_time=dimensionless_time)

Hamiltonian(f, mats::AbstractVector{T}; unit=:h, dimensionless_time=true, static=true) where {T<:SparseMatrixCSC} = SparseHamiltonian(f, mats, unit=unit, dimensionless_time=dimensionless_time)

Hamiltonian(mats; unit=:h, static=true) where {T<:Matrix} = ConstantHamiltonian(mats, unit=unit, static=static)