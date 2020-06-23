"""
$(TYPEDEF)

Defines open system operator set.

# Fields

$(FIELDS)
"""
struct OpenSysSet{T<:Tuple} <: AbstractOpenSys
    """Open system operators"""
    ops::T
end

function (O::OpenSysSet)(du, u, tf, t)
    for op in O.ops
        op(du, u, tf, t)
    end
end

function update_vectorized_cache!(cache, O::OpenSysSet, tf, t)
    for op in O.ops
        update_vectorized_cache!(cache, op, tf, t)
    end
end
