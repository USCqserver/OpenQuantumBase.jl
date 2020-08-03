Base.summary(annealing::AbstractAnnealing) = string(
    TYPE_COLOR,
    nameof(typeof(annealing)),
    NO_COLOR,
    " with hType ",
    TYPE_COLOR,
    typeof(annealing.H),
    NO_COLOR,
    " and uType ",
    TYPE_COLOR,
    typeof(annealing.u0),
    NO_COLOR,
)

function Base.show(io::IO, A::AbstractAnnealing)
    println(io, summary(A))
    print(io, "u0 with size: ")
    show(io, size(A.u0))
end

Base.summary(interaction::Interaction) = string(
    TYPE_COLOR,
    nameof(typeof(interaction)),
    NO_COLOR,
)

function Base.show(io::IO, I::Interaction)
    print(io, summary(I))
    print(io, " with ")
    show(io, I.coupling)
    println(io)
    print(io, "and bath: ")
    show(io, I.bath)
end

Base.summary(interactions::InteractionSet) = string(
    TYPE_COLOR,
    nameof(typeof(interactions)),
    NO_COLOR,
)

function Base.show(io::IO, I::InteractionSet)
    print(io, summary(I))
    print(io, " with ")
    show(io, length(I))
    print(io, " interactions.")
end
