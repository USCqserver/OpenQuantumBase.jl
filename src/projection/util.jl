struct LinearIdxLowerTriangular{T}
    mat::Matrix{T}
    lvl
    idx_exchange_func
end


function LinearIdxLowerTriangular(T, t_dim, lvl; idx_exchange_func = (x) -> x)
    s_dim = lvl * (lvl - 1) ÷ 2
    mat = Matrix{T}(undef, t_dim, s_dim)
    LinearIdxLowerTriangular(mat, lvl, idx_exchange_func)
end


function Base.getindex(m::LinearIdxLowerTriangular, i, j)
    getindex(m.mat, i, j)
end


function Base.setindex!(m::LinearIdxLowerTriangular, x, i, j, k)
    if j > k
        lin_idx = linear_idx_off(j, k, m.lvl)
        setindex!(m.mat, x, i, lin_idx)
    elseif j < k
        lin_idx = linear_idx_off(k, j, m.lvl)
        setindex!(m.mat, m.idx_exchange_func(x), i, lin_idx)
    else
        throw(ArgumentError("There is no diagonal element of this matrix type."))
    end
end


function Base.setindex!(m::LinearIdxLowerTriangular, x, i, j)
    setindex!(m.mat, x, i, j)
end


function Base.setindex!(m::LinearIdxLowerTriangular, x::T, i, j) where T<:Number
    m.mat[i, j] .= x
end


function Base.getindex(m::LinearIdxLowerTriangular, i, j, k)
    if j > k
        lin_idx = linear_idx_off(j, k, m.lvl)
        res = getindex(m.mat, i, lin_idx)
    elseif j < k
        lin_idx = linear_idx_off(k, j, m.lvl)
        res = getindex(m.mat, i, lin_idx)
        res = m.idx_exchange_func(res)
    else
        throw(ArgumentError("There is no diagonal element of this matrix type."))
    end
    res
end


function linear_idx_off(i, j, lvl)
    (2lvl - j) * (j - 1) ÷ 2 + i - j
end


Base.summary(A::LinearIdxLowerTriangular{T}) where T =
    string(
        TYPE_COLOR,
        nameof(typeof(A)),
        NO_COLOR,
        " with eType ",
        TYPE_COLOR,
        T,
        NO_COLOR
    )


function Base.show(io::IO, A::LinearIdxLowerTriangular)
    println(io, summary(A))
    print(io, "Size: ")
    show(io, (A.lvl, A.lvl))
end


function linear_idx(i, j, lvl)
    (2lvl - j + 2) * (j - 1) ÷ 2 + i - j + 1
end
