function hamiltonian_factory(funcs, ops; is_real=false)
    if is_real == true
        ops = real.(ops)
    end
    if issparse(ops[1])
        cache = spzeros(eltype(ops[1]), size(ops[1])...)
        operator = AffineOperatorSparse(funcs, ops)
        HamiltonianSparse(operator, cache)
    else
        cache = zeros(eltype(ops[1]), size(ops[1]))
        operator = AffineOperator(funcs, ops)
        Hamiltonian(operator, cache)
    end
end

function hamiltonian_factory(hfuncs, hops, gfuncs, gops)
    geometric_op = AffineOperator(gfuncs, gops)
    adiabatic_op = AffineOperator(hfuncs, hops)
    AdiabaticFrameHamiltonian(geometric_op, adiabatic_op,
    zeros(eltype(hops[1]), size(hops[1])),
    zeros(eltype(hops[1]), size(hops[1], 1)))
end
