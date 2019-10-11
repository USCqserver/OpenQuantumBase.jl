function construct_interpolations(
    x::AbstractRange{S},
    y::AbstractArray{T,1};
    method = BSpline(Quadratic(Line(OnGrid()))),
    extrapolation = "line",
) where {S<:Real,T<:Number}
    itp = interpolate(y, method)
    if lowercase(extrapolation) == "line"
        itp = extrapolate(itp, Line())
    end
    scale(itp, x)
end


function construct_interpolations(
    x::AbstractRange{S},
    y::AbstractArray{T,3};
    extrapolation = "line",
) where {S<:Real,T<:Number}
    i, j, = size(y)
    itp = interpolate(
        y,
        (NoInterp(), NoInterp(), BSpline(Quadratic(Line(OnGrid())))),
    )
    if lowercase(extrapolation) == "line"
        itp = extrapolate(itp, Line())
    end
    scale(itp, 1:i, 1:j, x)
end


function construct_interpolations(
    x::Vector{T},
    y::Vector{T};
    extrapolation = "line",
) where {T<:Real}
    itp = interpolate((x,), y, Gridded(Linear()))
    if lowercase(extrapolation) == "line"
        itp = extrapolate(itp, Line())
    end
    itp
end


function construct_interpolations(
    x::Vector{T},
    y::Array{T,3};
    extrapolation = "line",
) where {T<:Real}
    i, j, k = size(y)
    if i == 1
        knots = (1:j, x)
        y = y[1, :, :]
        itp = interpolate(knots, y, (NoInterp(), Gridded(Linear())))
    else
        knots = (1:i, 1:j, x)
        itp = interpolate(knots, y, (NoInterp(), NoInterp(), Gridded(Linear())))
    end
    if lowercase(extrapolation) == "line"
        itp = extrapolate(itp, Line())
    end
    itp
end
