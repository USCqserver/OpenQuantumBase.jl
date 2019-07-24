Base.summary(H::AbstractHamiltonian) = string(TYPE_COLOR, nameof(typeof(H)),
                                       NO_COLOR, " with ",
                                       TYPE_COLOR, typeof(H).parameters[1],
                                       NO_COLOR)

function Base.show(io::IO, A::AbstractHamiltonian)
    println(io, summary(A))
    print(io, "with size: ")
    show(io, A.size)
end
