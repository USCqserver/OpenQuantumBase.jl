struct ConstantCouplings
    mats
    str_rep
end

function (c::ConstantCouplings)(t)
    c.mats
end

function ConstantCouplings(mats::Union{Vector{Matrix{T}}, Vector{SparseMatrixCSC{T,Int}}}; str_rep = nothing) where T<:Number
    if str_rep !=nothing
        for s in str_rep
            if !(typeof(s) <: AbstractString)
                throw(ArgumentError("String representation can only be strings."))
            end
        end
    end
    ConstantCouplings(2π * mats, str_rep)
end

function ConstantCouplings(c::Vector{T}; sp = false) where T <: AbstractString
    mats = 2π * q_translate.(c, sp=sp)
    ConstantCouplings(mats, c)
end


Base.summary(C::ConstantCouplings) = string(TYPE_COLOR, nameof(typeof(C)),
                                       NO_COLOR, " with ",
                                       TYPE_COLOR, typeof(C.mats).parameters[1],
                                       NO_COLOR)


function Base.show(io::IO, C::ConstantCouplings)
    println(io, summary(C))
    print(io, "with string representation: ")
    show(io, C.str_rep)
end

struct TimeDependentCouplings
    funcs
    mats
end

function (c::TimeDependentCouplings)(t)
    [f(t) * m for (f, m) in zip(c.funcs, c.mats)]
end
