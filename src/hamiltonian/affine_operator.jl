"""
$(TYPEDEF)

Base for types defining dense linear operators.
"""
abstract type LinearOperator{T <: Number} end


"""
$(TYPEDEF)

Base for types defining sparse linear operators.
"""
abstract type LinearOperatorSparse{T <: Number} end


"""
$(TYPEDEF)

Defines an affine operator with dense matrices.

# Fields

$(FIELDS)
"""
struct AffineOperator{T <: Number} <: LinearOperator{T}
    " List of time dependent functions "
    f
    " List of constant matrices "
    m::Vector{Matrix{T}}

    AffineOperator(f, m) = all((x)->size(x) == size(m[1]), m) ? new{eltype(m[1])}(f, m) : throw(ArgumentError("Matrices in the list do not have the same size."))
end

"""
$(TYPEDEF)

Defines an affine operator with sparse matrices.

# Fields

$(FIELDS)
"""
struct AffineOperatorSparse{T <: Number} <: LinearOperatorSparse{T}
    " List of time dependent functions "
    f
    " List of constant matrices "
    m::Vector{SparseMatrixCSC{T,Int}}

    AffineOperatorSparse(f, m) = all((x)->size(x) == size(m[1]), m) ? new{eltype(m[1])}(f, m) : throw(ArgumentError("Matrices in the list do not have the same size."))
end


function (A::AffineOperator)(du, p, t)
    for (f, m) in zip(A.f, A.m)
        axpy!(p * f(t), m, du)
    end
end


function (A::AffineOperator)(du, t)
    for (f, m) in zip(A.f, A.m)
        axpy!(f(t), m, du)
    end
end


function (A::AffineOperatorSparse)(du, t)
    for (f, m) in zip(A.f, A.m)
        du .+= f(t) * m
    end
end


function (A::AffineOperatorSparse)(du, p, t)
    for (f, m) in zip(A.f, A.m)
        du .+= p * f(t) * m
    end
end
