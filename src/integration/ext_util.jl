"""
$(TYPEDEF)

A tag for inplace unitary function. `func(cache, t)` is the actual inplace update function.

# Fields

$(FIELDS)
"""
struct InplaceUnitary
    "inplace update function"
    func::Any
end

isinplace(::Any) = false
isinplace(::InplaceUnitary) = true

"""
$(SIGNATURES)

Calculate the Lamb shift of spectrum `γ` at angular frequency `ω`. All keyword arguments of `quadgk` function is supported.
"""
function lambshift(ω, γ; kwargs...)
    integrand = (x)->(γ(ω+x) - γ(ω-x))/x
    integral, = quadgk(integrand, 0, Inf; kwargs...)
    # TODO: do something with the error information
    -integral / 2 / π
end
