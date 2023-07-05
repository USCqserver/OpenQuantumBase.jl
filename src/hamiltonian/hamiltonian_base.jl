"""
$(SIGNATURES)

Evaluates a time-dependent Hamiltonian at time `s`, expressed in units of `GHz`.
For generic `AbstractHamiltonian` types, it defaults to `H.(s)/2/π`.
"""
evaluate(H::AbstractHamiltonian, s::Real) = H.(s) / 2 / π

"""
$(SIGNATURES)

This function provides a generic `evaluate` interface for `AbstractHamiltonian`
types that accepts two arguments. It ensures that other concrete Hamiltonian 
types behave consistently with methods designed for `AdiabaticFrameHamiltonian`.
"""
evaluate(H::AbstractHamiltonian, ::Any, s::Real) = H.(s) / 2 / π

"""
    isconstant(H)

Verifies if a Hamiltonian is constant. By default, it returns false for a generic
Hamiltonian.
"""
isconstant(::AbstractHamiltonian) = false

"""
    issparse(H)

Verifies if a Hamiltonian is sparse. By default, it returns false for a generic
Hamiltonian.
"""
issparse(::AbstractHamiltonian) = false 

Base.eltype(::AbstractHamiltonian{T}) where {T} = T
Base.size(H::AbstractHamiltonian) = H.size
Base.size(H::AbstractHamiltonian, dim::T) where {T<:Integer} = H.size[dim]
get_cache(H::AbstractHamiltonian) = H.u_cache

"""
$(SIGNATURES)

Update the internal cache `cache` according to the value of the Hamiltonian `H` 
at given time `t`: ``cache = -iH(p, t)``. The third argument, `p` 
is reserved for passing additional info to the `AbstractHamiltonian` object. 
Currently, it is only used by `AdiabaticFrameHamiltonian` to pass the total 
evolution time `tf`. To keep the interface consistent across all 
`AbstractHamiltonian` types, the `update_cache!` method for all subtypes of 
`AbstractHamiltonian` should keep the argument `p`.

Fallback to `cache .= -1.0im * H(p, t)` for generic `AbstractHamiltonian` type.
"""
update_cache!(cache, H::AbstractHamiltonian, p, t::Real) = cache .= -1.0im * H(p, t)

"""
$(SIGNATURES)

This function calculates the vectorized version of the commutation relation 
between the Hamiltonian `H` at time `t` and the density matrix ``ρ``, and then 
updates the cache in-place.

The commutation relation is given by ``[H, ρ] = Hρ - ρH``. The vectorized 
commutator is given by ``I⊗H-H^T⊗I``. 

...
# Arguments
- `cache`: the variable to be updated in-place, storing the vectorized commutator.
- `H::AbstractHamiltonian`: an instance of AbstractHamiltonian, representing the Hamiltonian of the system.
- `p`: unused parameter, kept for function signature consistency with other methods.
- `t`: a real number representing the time at which the Hamiltonian is evaluated.

# Returns
The function does not return anything as the update is performed in-place on cache.
...
"""
function update_vectorized_cache!(cache, H::AbstractHamiltonian, p, t::Real)
    hmat = H(t)
    iden = one(hmat)
    cache .= 1.0im * (transpose(hmat) ⊗ iden - iden ⊗ hmat)
end

"""
$(SIGNATURES)

This function implements the Liouville-von Neumann equation, describing the time
evolution of a quantum system governed by the Hamiltonian `h`.

The Liouville-von Neumann equation is given by ``du/dt = -i[H, ρ]``, where ``H``
is the Hamiltonian of the system, ``ρ`` is the density matrix (`u` in this 
context), and ``[H, ρ]`` is the commutation relation between ``H`` and ``ρ``, 
defined as ``Hρ - ρH``.

The function is written in such a way that it can be directly passed to 
differential equation solvers in Julia, such as those in 
`DifferentialEquations.jl`, as the system function representing the ODE to be solved.

...
# Arguments
- `du`: the derivative of the density matrix with respect to time. The result 
        of the calculation will be stored here.
- `u`: an instance of `AbstractMatrix`, representing the density matrix of the system.
- `p`: unused parameter, kept for function signature consistency with other methods.
- `t`: a real number representing the time at which the Hamiltonian is evaluated.

# Returns
The function does not return anything as the update is performed in-place on `du`.
...
"""
function (h::AbstractHamiltonian)(du, u::AbstractMatrix, p, t::Real)
    H = h(t)
    Hρ = -1.0im * H * u
    du .= Hρ - transpose(Hρ)
end

"""
$(SIGNATURES)

The `AbstractHamiltonian` type can be called with two arguments. The first 
argument is reserved to pass additional info to the `AbstractHamiltonian` object. 
Currently, it is only used by `AdiabaticFrameHamiltonian` to pass the total 
evolution time `tf`.

Fallback to `H(t)` for generic `AbstractHamiltonian` type.
"""
(H::AbstractHamiltonian)(::Any, t::Real) = H(t)

Base.summary(H::AbstractHamiltonian) = string(
    TYPE_COLOR,
    nameof(typeof(H)),
    NO_COLOR,
    " with ",
    TYPE_COLOR,
    typeof(H).parameters[1],
    NO_COLOR,
)

function Base.show(io::IO, A::AbstractHamiltonian)
    println(io, summary(A))
    print(io, "with size: ")
    show(io, size(A))
end

"""
$(SIGNATURES)

Default eigenvalue decomposition method for an abstract Hamiltonian `H` at
time `t`. Keyword argument `lvl` specifies the number of levels to keep in 
the output. 

The function returns a tuple (w, v), where `w` is a vector of eigenvalues, 
and `v` is a matrix where each column represents an eigenvector. (The `k`th 
eigenvector can be extracted using the slice `v[:, k]`.)
"""
# If H(t) returns an array, the `Array` function will not allocate a new 
# variable
haml_eigs_default(H::AbstractHamiltonian, t, lvl::Integer) = eigen!(
    H(t) |> Array |> Hermitian, 1:lvl)
haml_eigs_default(H::AbstractHamiltonian, t, ::Nothing) = eigen(
    H(t) |> Array |> Hermitian)
haml_eigs(H::AbstractHamiltonian, t, lvl; kwargs...) = haml_eigs_default(H, t,
    lvl; kwargs...)

#function eigen!(M::Hermitian{T, S}, lvl::UnitRange) where T<:Number where S<:Union{SMatrix, MMatrix}
#    w, v = eigen(Hermitian(M))
#    w[lvl], v[:, lvl]
#end

"""
$(SIGNATURES)

Calculate the eigen value decomposition of the Hamiltonian `H` at time `t`. 
Keyword argument `lvl` specifies the number of levels to keep in the output. 
`w` is a vector of eigenvalues and `v` is a matrix of the eigenvectors in the 
columns. (The `k`th eigenvector can be obtained from the slice `v[:, k]`.) `w` 
will be in unit of `GHz`.
"""
function eigen_decomp(H::AbstractHamiltonian, t::Real; lvl::Int=2, kwargs...)
    w, v = haml_eigs(H, t, lvl; kwargs...)
    real(w)[1:lvl] / 2 / π, v[:, 1:lvl]
end

eigen_decomp(H::AbstractHamiltonian; lvl::Int=2, kwargs...) = isconstant(H) ? 
    eigen_decomp(H, 0, lvl=lvl, kwargs...) : throw(ArgumentError("H must be a constant Hamiltonian"))

"""
$(SIGNATURES)

Calculate the eigen value decomposition of the Hamiltonian `H` at an array of 
time points `s`. The output keeps the lowest `lvl` eigenstates and their 
corresponding eigenvalues. Output `(vals, vecs)` have the dimensions of 
`(lvl, length(s))` and `(size(H, 1), lvl, length(s))` respectively.
"""
function eigen_decomp(
    H::AbstractHamiltonian,
    s::AbstractArray{Float64,1};
    lvl::Int=2,
    kwargs...
)
    s_dim = length(s)
    res_val = Array{eltype(H),2}(undef, (lvl, s_dim))
    res_vec = Array{eltype(H),3}(undef, (size(H, 1), lvl, s_dim))
    for (i, s_val) in enumerate(s)
        val, vec = haml_eigs(H, s_val, lvl; kwargs...)
        res_val[:, i] = val[1:lvl]
        res_vec[:, :, i] = vec[:, 1:lvl]
    end
    res_val, res_vec
end

"""
$(SIGNATURES)

For a time series quantum states given by `states`, whose time points are given
by `s`, calculate the population of instantaneous eigenstates of `H`. The levels
of the instantaneous eigenstates are specified by `lvl`, which can be any slice index.
"""
function inst_population(s, states, H::AbstractHamiltonian; lvl=1:1)
    if typeof(lvl) <: Int
        lvl = lvl:lvl
    end
    pop = Array{Array{Float64,1},1}(undef, length(s))
    for (i, v) in enumerate(s)
        w, v = eigen_decomp(H, v, lvl=maximum(lvl))
        if ndims(states[i]) == 1
            inst_state = view(v, :, lvl)'
            pop[i] = abs2.(inst_state * states[i])
        elseif ndims(states[i]) == 2
            l = length(lvl)
            temp = Array{Float64,1}(undef, l)
            for j in range(1, length=l)
                inst_state = view(v, :, j)
                temp[j] = real(inst_state' * states[i] * inst_state)
            end
            pop[i] = temp
        end
    end
    pop
end

function is_complex(f_list, m_list)
    any(m_list) do m
        eltype(m) <: Complex
    end || any(f_list) do f
        typeof(f(0)) <: Complex
    end
end

