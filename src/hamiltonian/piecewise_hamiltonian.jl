struct PiecewiseHamiltonian{T, S<:AbstractHamiltonian{T}} <: AbstractHamiltonian{T}
    HSet::Vector{S}
    tstops
end

function PiecewiseHamiltonian(hset...; tstops=[])
    if !all((x)->typeof(x)==typeof(hset[1]), hset)
        throw(TypeError("All the Hamiltonians must have the same type."))
    end
    PiecewiseHamiltonian([h for h in hset], tstops)
end
