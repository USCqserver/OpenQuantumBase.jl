"""
$(TYPEDEF)

Defines rates of Davies form ME acting on the diagonal elements.

# Fields

$(FIELDS)
"""
struct ΓMatrix
    """interpolated Γ matrix"""
    m
    """levels of the system"""
    lvl
end


function ΓMatrix(t_idx, data::Array{T,3}) where {T<:Real}
    i, j, k = size(data)
    if i != j - 1
        throw(ArgumentError("The first dimension shoud be level-1."))
    end
    itp = construct_interpolations(t_idx, data)
    if i == 1
        itp = Γ2D(itp)
    end
    ΓMatrix(itp, j)
end


function (G::ΓMatrix)(t)
    lvl = G.lvl
    Γ = zeros(lvl, lvl)
    for j = 1:lvl
        for i = 1:(j-1)
            Γ[i, j] = G.m(i, j, t)
        end
        for i = j:(lvl-1)
            Γ[i+1, j] = G.m(i, j, t)
        end
    end
    diag_elem = -sum(Γ, dims = 1)
    for i = 1:lvl
        Γ[i, i] = diag_elem[i]
    end
    Γ
end


"""
$(TYPEDEF)

Defines two-level Γ matrix. This object will be obselte when interpolate.jl support singleton demension.

# Fields

$(FIELDS)
"""
struct Γ2D
    itp
end


"""
    function (G::Γ2D)(i, j, k)

When calling object Γ2D, the first index will with ignored.
"""
function (G::Γ2D)(i, j, k)
    G.itp(j, k)
end
