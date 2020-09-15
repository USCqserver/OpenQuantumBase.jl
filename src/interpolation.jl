import Interpolations:
    interpolate,
    BSpline,
    Quadratic,
    Line,
    OnGrid,
    scale,
    gradient1,
    extrapolate,
    Linear,
    Gridded,
    NoInterp,
    Cubic,
    Constant,
    Flat

"""
$(SIGNATURES)

Construct interpolation of N-D array `y` along its last dimension on grid `x`. `method` specifies the interpolation algorithm, which can be either "BSpline" or "Gridded". If `x` is an `AbstractRange`, the default method is "BSpline", otherwise the default `method` is "Gridded". `order` is the interpolation order. If `x` is `AbstractRange`, the default order is 2, otherwise the default order is 1. `extrapolation` specifies the extrapolation methods.
"""
function construct_interpolations(
    x::AbstractRange{S},
    y::AbstractArray{T,N};
    method="BSpline",
    order=2,
    extrapolation="line",
) where {S <: Real,T <: Number,N}
    method = interp_tuple(N, method, order)
    itp = interpolate(y, method)
    if lowercase(extrapolation) == "line"
        itp = extrapolate(itp, Line())
    elseif lowercase(extrapolation) == "flat"
        itp = extrapolate(itp, Flat())
    end
    index = interp_index(x, size(y))
    scale(itp, index...)
end


function construct_interpolations(
    x::AbstractArray{S},
    y::AbstractArray{T,N};
    method="Gridded",
    order=1,
    extrapolation="line",
) where {S <: Real,T <: Number,N}
    if lowercase(method) == "bspline"
        Δ = diff(x)
        if allequal(Δ)
            x = range(x[1], x[end], length=length(x))
            return construct_interpolations(
                x,
                y,
                method=method,
                order=order,
                extrapolation=extrapolation,
            )
        else
            @warn "The grid is not uniform. Using grided linear interpolation."
            method = "Gridded"
        end
    end
    method = interp_tuple(N, method, order)
    index = interp_index(x, size(y))
    itp = interpolate(index, y, method)
    if lowercase(extrapolation) == "line"
        itp = extrapolate(itp, Line())
    elseif lowercase(extrapolation) == "flat"
        itp = extrapolate(itp, Flat())
    end
    itp
end


function construct_interpolations(
    x::AbstractRange{S},
    y::AbstractArray{W,1};
    method="BSpline",
    order=2,
    extrapolation="line",
) where {S <: Real,W <: AbstractArray}
    y = cat(y...; dims=ndims(y[1]) + 1)
    construct_interpolations(
        x,
        y,
        method=method,
        order=order,
        extrapolation=extrapolation,
    )
end

"""
$(SIGNATURES)

Construct interpolation of a nested array. If `x` is an `AbstractRange`, the nested array will be converted into multi-dimensional array for interpolation. Otherwise only "Gridded" method and `order` 1 is supported.
"""
function construct_interpolations(
    x::AbstractArray{S,1},
    y::AbstractArray{W,1};
    method="Gridded",
    order=1,
    extrapolation="line",
) where {S <: Real,W <: AbstractArray}
    itp = interpolate((x,), y, interp_method(method, order))
    if lowercase(extrapolation) == "line"
        itp = extrapolate(itp, Line())
    elseif lowercase(extrapolation) == "flat"
        itp = extrapolate(itp, Flat())
    end
    itp
end


function interp_index(x, s)
    res = []
    for i = 1:(length(s) - 1)
        push!(res, 1:s[i])
    end
    push!(res, x)
    (res...,)
end


function interp_tuple(N, method, order)
    res = []
    for i = 1:(N - 1)
        push!(res, NoInterp())
    end
    interp_term = interp_method(method, order)
    push!(res, interp_term)
    (res...,)
end


function interp_method(method, order::Integer; boundary=Line(OnGrid()))
    if lowercase(method) == "bspline"
        if order == 0
            res = BSpline(Constant())
        elseif order == 1
            res = BSpline(Linear())
        elseif order == 2
            res = BSpline(Quadratic(boundary))
        elseif order == 3
            res = BSpline(Cubic(boundary))
        else
            throw(ArgumentError("Order $order for method $method is not implemented."))
        end
    elseif lowercase(method) == "gridded"
        if order == 0
            res = Gridded(Constant())
        elseif order == 1
            res = Gridded(Linear())
        else
            throw(ArgumentError("Oder $order for method $method is not implemented."))
        end
    elseif lowercase(method) == "none"
        res = NoInterp()
    else
        throw(ArgumentError("Method $method is invalid."))
    end
    res
end


"""
$(SIGNATURES)

Calculate the gradient of `itp` at `s`.
"""
function gradient(itp, s::Number)
    gradient1(itp, s)
end


function gradient(itp, s::AbstractArray)
    [gradient1(itp, x) for x in s]
end
