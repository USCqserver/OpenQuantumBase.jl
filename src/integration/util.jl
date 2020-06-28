"""
$(TYPEDEF)

A tag for inplace unitary function. `func(cache, t)` is the actual inplace update function.

# Fields

$(FIELDS)
"""
struct InplaceUnitary
    """inplace update function"""
    func::Any
end

isinplace(::Any) = false
isinplace(::InplaceUnitary) = true
