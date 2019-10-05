struct Complex_Interp
    real
    imag
end

function (itp::Complex_Interp)(t)
    itp.real(t) + 1.0im * itp.imag(t)
end

struct Interp_Vector{T<:Number}
    itp::Vector{AbstractArray{T,1}}
end

struct Interp_Matrix{T<:Number}
    itp::Matrix{AbstractArray{T,1}}
end

function (Itp::Interp_Vector)(t)
    [x(t) for x in Itp.itp]
end

function construct_interpolations(
    x::AbstractRange{S},
    y::Vector{T};
    extrapolation = "line",
) where {S<:Real,T<:Complex}
    re = real(y)
    im = imag(y)
    re_itp = interpolate(re, BSpline(Quadratic(Line(OnGrid()))))
    im_itp = interpolate(im, BSpline(Quadratic(Line(OnGrid()))))
    if lowercase(extrapolation) == "line"
        re_itp = extrapolate(re_itp, Line())
        im_itp = extrapolate(im_itp, Line())
    end
    Complex_Interp(scale(re_itp, x), scale(im_itp, x))
end

function construct_interpolations(
    x::AbstractRange{S},
    y::AbstractArray{T,1};
    extrapolation = "line",
) where {S<:Real,T<:Real}
    itp = interpolate(y, BSpline(Quadratic(Line(OnGrid()))))
    if lowercase(extrapolation) == "line"
        itp = extrapolate(itp, Line())
    end
    scale(itp, x)
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

function construct_interpolations(x, y::Array{T,3}; extrapolation = "line") where {T<:Real}
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
