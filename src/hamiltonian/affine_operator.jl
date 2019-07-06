struct AffineOperator{T<:Number} <: LinearOperator{T}
    " List of time dependent functions "
    f
    " List of constant matrices "
    m::Vector{Matrix{T}}
end

struct AffineOperatorSparse{T<:Number} <: LinearOperatorSparse{T}
    " List of time dependent functions "
    f
    " List of constant matrices "
    m::Vector{SparseMatrixCSC{T, Int64}}
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

function (A::AffineOperatorSparse)(du, t)
    for (f, m) in zip(A.f, A.m)
        du .+= f(t)*m
    end
end

function (A::AffineOperatorSparse)(du, p, t)
    for (f, m) in zip(A.f, A.m)
        du .+= p*f(t)*m
    end
end
