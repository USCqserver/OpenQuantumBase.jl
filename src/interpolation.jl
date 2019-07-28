struct Complex_Interp
    real
    imag
end

function (itp::Complex_Interp)(t)
    itp.real(t) + 1.0im*itp.imag(t)
end

struct Interp_Vector{T<:Number}
    itp::Vector{AbstractArray{T, 1}}
end

struct Interp_Matrix{T<:Number}
    itp::Matrix{AbstractArray{T, 1}}
end

function (Itp::Interp_Vector)(t)
    [x(t) for x in Itp.itp]
end

function construct_interpolations(x::AbstractRange{S}, y::Vector{T}; extrapolation="line") where {S<:Real, T<:Complex}
    re = real(y)
    im = imag(y)
    re_itp = interpolate(re,  BSpline(Quadratic(Line(OnGrid()))))
    im_itp = interpolate(im,  BSpline(Quadratic(Line(OnGrid()))))
    if lowercase(extrapolation) == "line"
        re_itp = extrapolate(re_itp, Line())
        im_itp = extrapolate(im_itp, Line())
    end
    Complex_Interp(scale(re_itp, x), scale(im_itp, x))
end

function construct_interpolations(x::AbstractRange{S}, y::AbstractArray{T, 1}; extrapolation="line") where {S<:Real, T<:Real}
    itp = interpolate(y,  BSpline(Quadratic(Line(OnGrid()))))
    if lowercase(extrapolation) == "line"
        itp = extrapolate(itp, Line())
    end
    scale(itp, x)
end
