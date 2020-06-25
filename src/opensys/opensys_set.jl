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

OpenSysSet(opensys_ops...) = OpenSysSet(opensys_ops)

function update_ρ!(du, u, p, t, O::OpenSysSet)
    for op in O.ops
        update_ρ!(du, u, p, t, op)
    end
end

function update_cache!(cache, u, p, t, O::OpenSysSet)
    for op in O.ops
        update_cache!(cache, u, p, t, op)
    end
end

function update_vectorized_cache!(cache, u, p, t, O::OpenSysSet)
    for op in O.ops
        update_vectorized_cache!(cache, u, p, t, op)
    end
end
