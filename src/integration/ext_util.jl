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

Calculate the Lamb shift of spectrum `γ`. `atol` is the absolute tolerance for Cauchy principal value integral.
"""
function lambshift_cpvagk(w, γ; atol = 1e-7)
    g(x) = γ(x) / (x - w)
    cpv, cperr = cpvagk(γ, w, w - 1.0, w + 1.0)
    negv, negerr = quadgk(g, -Inf, w - 1.0)
    posv, poserr = quadgk(g, w + 1.0, Inf)
    v = cpv + negv + posv
    err = cperr + negerr + poserr
    if (err > atol) || (isnan(err))
        @warn "Absolute error of integration is larger than the tolerance."
    end
    -v / 2 / pi
end
