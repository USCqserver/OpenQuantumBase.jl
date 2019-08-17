struct ConstantCouplings
    mats
    str_rep
end

function (c::ConstantCouplings)(t)
    c.mats
end

Base.iterate(c::ConstantCouplings, state=1) = Base.iterate(c.mats, state)

Base.length(c::ConstantCouplings) = length(c.mats)

Base.eltype(c::ConstantCouplings) = typeof(c.mats[1])

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


struct TimeDependentCoupling
    funcs
    mats
    TimeDependentCoupling(funcs, mats) = new(funcs, 2π*mats)
end


function (c::TimeDependentCoupling)(t)
    sum((x)->x[1](t)*x[2], zip(c.funcs, c.mats))
end


struct TimeDependentCouplings
    coupling::Tuple
end


function TimeDependentCouplings(args...)
    TimeDependentCouplings(args)
end


function (c::TimeDependentCouplings)(t)
    [x(t) for x in c.coupling]
end


Base.iterate(c::TimeDependentCouplings, state=1) = Base.iterate(c.coupling, state)

Base.length(c::TimeDependentCouplings) = length(c.coupling)

Base.eltype(c::TimeDependentCouplings) = typeof(c.coupling[1])
