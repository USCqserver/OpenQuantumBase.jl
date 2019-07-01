struct AffineOperator{T<:Number} <: LinearOperator{T}
    " List of time dependent functions "
    f
    " List of constant matrices "
    m::Union{Array{Array{T, 2}, 1}, Array{SparseMatrixCSC{T, Int64}, 1}}
end

function (A::AffineOperator)(du, p, t)
    for (f, m) in zip(A.f, A.m)
        axpy!(p*f(t), m, du)
    end
end

function (A::AffineOperator)(du, t)
    for (f, m) in zip(A.f, A.m)
        axpy!(f(t), m, du)
    end
end
