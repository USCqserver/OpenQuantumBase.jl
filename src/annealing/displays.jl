Base.summary(annealing::AbstractAnnealing) = string(TYPE_COLOR, nameof(typeof(annealing)),
                                       NO_COLOR, " with hType ",
                                       TYPE_COLOR, typeof(annealing.H),
                                       NO_COLOR, " and uType ",
                                       TYPE_COLOR, typeof(annealing.u0),
                                       NO_COLOR)

function Base.show(io::IO, A::AbstractAnnealing)
    println(io, summary(A))
    print(io, "s parameter span: ")
    show(io, A.sspan)
    println(io)
    print(io, "u0: ")
    show(io, A.u0)
    println(io)
    print(io, "bath: ")
    show(io, A.bath)
end
