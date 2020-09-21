"""
$(TYPEDEF)

Object for a projected low level system. The projection is only valid for real Hamiltonians.

# Fields

$(FIELDS)
"""
struct ProjectedSystem
    """Time grid (unitless) for projection"""
    s::AbstractArray{Float64,1}
    """Energy values for different levels"""
    ev::Array{Vector{Float64},1}
    """Geometric terms"""
    dθ::Array{Vector{Float64},1}
    """Projected system bath interaction operators"""
    op::Array{Array{Matrix{Float64},1},1}
    """Direction for the calculation"""
    direct::Symbol
    """Number of leves to keep"""
    lvl::Int
    """Energy eigenstates at the final time"""
    ref::Array{Float64,2}
end

function ProjectedSystem(s, lvl, direction, ref)
    len = length(s)
    ev = Vector{Vector{Float64}}()
    dθ = Vector{Vector{Float64}}()
    op = Vector{Vector{Matrix{Float64}}}()
    ProjectedSystem(s, ev, dθ, op, direction, lvl, ref)
end

"""
$(SIGNATURES)

Project a Hamiltonian `H` to the lowest `lvl` level subspace. `s_axis` is the grid of (unitless) times on which the projection is calculated. `dH` is the derivative of the Hamiltonian. `coupling` is the system-bath interaction operator. Both of `coupling` and `dH` should be callable with annealing parameter `s`. `atol` and `rtol` are the absolute and relative error tolerance to distinguish two degenerate energy levels. `direction`, which can be either `:forward` or `backward`, controls whether to start the calcuation from the starting point or the end point. Currently this function only support real Hamiltonian with non-degenerate energies.
"""
function project_to_lowlevel(
    H::AbstractHamiltonian{T},
    s_axis::AbstractArray{S,1},
    coupling,
    dH;
    lvl=2,
    atol::Real=1e-6,
    rtol::Real=0,
    direction=:forward,
    refs=zeros(0, 0),
) where {T <: Real,S <: Real}

    if direction == :forward
        _s_axis = s_axis
        update_rule = push!
    elseif direction == :backward
        _s_axis = reverse(s_axis)
        update_rule = pushfirst!
    else
        throw(ArgumentError("direction $direction is not supported."))
    end
    if isempty(refs)
        w, v = H.EIGS(H, _s_axis[1], lvl)
        # this is needed for StaticArrays
        w = w[1:lvl]
        v = v[:, 1:lvl]
        d_inds = find_degenerate(w, atol=atol, rtol=rtol)
        if !isempty(d_inds)
            @warn "Degenerate energy levels detected at" _s_axis[1]
            @warn "With" d_inds
        end
        refs = Array(v)
        projected_system = ProjectedSystem(s_axis, lvl, direction, refs)
        update_params!(projected_system, w, dH(_s_axis[1]), coupling(_s_axis[1]), update_rule, d_inds)
        _s_axis = _s_axis[2:end]
    else
        projected_system = ProjectedSystem(s_axis, lvl, direction, refs)
    end

    for s in _s_axis
        w, v = H.EIGS(H, s, lvl)
        # this is needed for StaticArrays
        w = w[1:lvl]
        v = v[:, 1:lvl]
        d_inds = find_degenerate(w, atol=atol, rtol=rtol)
        if !isempty(d_inds)
            @warn "Possible degenerate detected at" s
            @warn "With levels" d_inds
        end
        update_refs!(projected_system, v, lvl, d_inds)
        update_params!(projected_system, w, dH(s), coupling(s), update_rule, d_inds)
    end
    projected_system
end

function project_to_lowlevel(
    H::AbstractHamiltonian{T},
    s_axis::AbstractArray{S,1},
    coupling,
    dH;
    lvl=2,
    atol::Real=1e-6,
    rtol::Real=0,
    direction=:forward,
    refs=zeros(0, 0),
) where {T <: Complex,S <: Real}
    @warn "The projection method only works with real Hamitonians. Convert the complex Hamiltonian to real one."
    H_real = convert(Real, H)
    project_to_lowlevel(H_real, s_axis, coupling, dH, lvl=lvl, atol=atol, rtol=rtol, direction=direction, refs=refs)
end

function update_refs!(refs, v, lvl, d_inds)
    # update reference vectors for degenerate subspace
    if !isempty(d_inds)
        for inds in d_inds
            v[:, inds] = (v[:, inds] / refs[:, inds]) * v[:, inds]
        end
        flat_d_inds = reduce(vcat, d_inds)
    else
        flat_d_inds = []
    end
    # update reference vectors for non-degenerate states
    for i in (k for k in 1:lvl if !(k in flat_d_inds))
    #for i in 1:lvl
        if v[:, i]' * refs[:, i] < 0
            refs[:, i] = -v[:, i]
        else
            refs[:, i] = v[:, i]
        end
    end
end

update_refs!(sys::ProjectedSystem, v, lvl, d_inds) = update_refs!(sys.ref, v, lvl, d_inds)

function update_params!(sys::ProjectedSystem, w, dH, interaction, update_rule, d_inds)
    # update energies
    E = w / 2 / π
    update_rule(sys.ev, E)
    # update dθ
    dθ = Vector{Float64}()
    for j = 1:sys.lvl
        for i = (j + 1):sys.lvl
            if any((x) -> issubset([i,j], x), d_inds)
                # for degenerate levels, push in 0 for now
                push!(dθ, 0.0)
            else
                vi = @view sys.ref[:, i]
                vj = @view sys.ref[:, j]
                t = vi' * dH * vj / (E[j] - E[i])
                push!(dθ, t)
            end
        end
    end
    update_rule(sys.dθ, dθ)
    # update projected interaction operators
    op = [sys.ref' * x * sys.ref for x in interaction]
    update_rule(sys.op, op)
end

"""
    get_dθ(sys::ProjectedSystem, i=1, j=2)

Get the geometric terms between i, j energy levels from `ProjectedSystem`.
"""
function get_dθ(sys::ProjectedSystem, i=1, j=2)
    if j > i
        idx = (2 * sys.lvl - i) * (i - 1) ÷ 2 + (j - i)
        return [-x[idx] for x in sys.dθ]
    elseif j < i
        idx = (2 * sys.lvl - j) * (j - 1) ÷ 2 + (i - j)
        return [x[idx] for x in sys.dθ]
    else
        error("No diagonal element for dθ.")
    end
end


"""
    function concatenate(args...)

Concatenate multiple `ProjectedSystem` objects into a single one. The arguments need to be in time order. The `ref` field of the new object will have the same value as the last input arguments.
"""
function concatenate(args...)
    s = vcat([sys.s for sys in args]...)
    ev = vcat([sys.ev for sys in args]...)
    dθ = vcat([sys.dθ for sys in args]...)
    op = vcat([sys.op for sys in args]...)
    ref = args[end].ref
    lvl = args[end].lvl
    ProjectedSystem(s, ev, dθ, op, ref, lvl)
end