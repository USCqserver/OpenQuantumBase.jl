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
    """Energy eigenstates at the final time"""
    ref::Array{Float64,2}
    """Number of levels being kept"""
    lvl::Int
end


"""
$(TYPEDEF)

Object for a projected coupling parameterized by NIBA parameters. This object holds the numerical values at gridded points.

# Fields

$(FIELDS)
"""
struct ProjectedCoupling
    """Time grid (unitless) for projection"""
    s::Any
    """``(σ_m - σ_n)^2``"""
    a::Any
    """``|σ_{mn}|^2``"""
    b::Any
    """``σ_{mn}(σ_m-σ_n)``"""
    c::Any
    """``σ_{mn}(σ_m+σ_n)``"""
    d::Any
end


"""
$(TYPEDEF)

Object for a projected system Hamiltonian parameterized by NIBA parameters. This object holds the numerical values at gridded points.

# Fields

$(FIELDS)
"""
struct ProjectedTG
    """Time grid (unitless) for projection"""
    s::Any
    """Frequencies of given basis states"""
    ω::Any
    """Adiabatic part"""
    T::Any
    """Geometric phase"""
    G::Any
end


function ProjectedSystem(s, ref, lvl)
    len = length(s)
    ev = Vector{Vector{Float64}}()
    dθ = Vector{Vector{Float64}}()
    op = Vector{Vector{Matrix{Float64}}}()
    ProjectedSystem(s, ev, dθ, op, ref, lvl)
end


function ProjectedCoupling(s, lvl)
    t_dim = length(s)
    a = LinearIdxLowerTriangular(Float64, t_dim, lvl)
    b = LinearIdxLowerTriangular(Float64, t_dim, lvl)
    c = LinearIdxLowerTriangular(
        Float64,
        t_dim,
        lvl;
        idx_exchange_func=(x) -> -conj(x),
    )
    d = LinearIdxLowerTriangular(
        Float64,
        t_dim,
        lvl;
        idx_exchange_func=(x) -> conj(x),
    )
    ProjectedCoupling(s, a, b, c, d)
end


function ProjectedTG(s, lvl)
    t_dim = length(s)
    ω = Matrix{Float64}(undef, t_dim, lvl)
    T = LinearIdxLowerTriangular(
        Float64,
        t_dim,
        lvl;
        idx_exchange_func=(x) -> x,
    )
    G = LinearIdxLowerTriangular(
        Float64,
        t_dim,
        lvl;
        idx_exchange_func=(x) -> -x,
    )
    ProjectedTG(s, ω, T, G)
end


"""
$(SIGNATURES)

Project a Hamiltonian `H` to the lowest `lvl` level subspace. `s_axis` is the grid of (unitless) times on which the projection is calculated. `dH` is the derivative of the Hamiltonian. `coupling` is the system-bath interaction operator. Both of `coupling` and `dH` should be callable with annealing parameter `s`. `atol` and `rtol` are the absolute and relative error tolerance to distinguish two degenerate energy levels. Currently this function only support real Hamiltonian with non-degenerate energies.
"""
function project_to_lowlevel(
    H::AbstractHamiltonian{T},
    s_axis::AbstractArray{S,1},
    coupling,
    dH;
    lvl=2,
    atol::Real=1e-6,
    rtol::Real=0,
    refs=zeros(0, 0),
) where {T <: Real,S <: Real}

    if isempty(refs)
        w, v = H.EIGS(H, s_axis[1], lvl)
        # this is needed for StaticArrays
        w = w[1:lvl]
        v = v[:, 1:lvl]
        d_inds = find_degenerate(w, atol=atol, rtol=rtol)
        if !isempty(d_inds)
            @warn "Possible degenerate detected at" s_axis[1]
            @warn "With levels" d_inds
        end
        refs = Array(v)
        rotate_refs!(refs, v, lvl)
        projected_system = ProjectedSystem(s_axis, refs, lvl)
        push_params!(projected_system, w, v, dH(s_axis[1]), coupling(s_axis[1]), d_inds)
        s_axis = s_axis[2:end]
    else
        projected_system = ProjectedSystem(s_axis, refs, lvl)
    end

    for s in s_axis
        w, v = H.EIGS(H, s, lvl)
        # this is needed for StaticArrays
        w = w[1:lvl]
        v = v[:, 1:lvl]
        d_inds = find_degenerate(w, atol=atol, rtol=rtol)
        if !isempty(d_inds)
            @warn "Possible degenerate detected at" s
            @warn "With levels" d_inds
        end
        rotate_refs!(projected_system, v, lvl)
        push_params!(projected_system, w, v, dH(s), coupling(s), d_inds)
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
    refs=zeros(0, 0),
) where {T <: Complex,S <: Real}
    @warn "The projection method only works with real Hamitonians. Convert the complex Hamiltonian to real one."
    H_real = convert(Real, H)
    project_to_lowlevel(H_real, s_axis, coupling, dH, lvl=lvl, atol=atol, rtol=rtol, refs=refs)
end

function rotate_refs!(refs, v, lvl)
    # update reference vectors for non-degenerate states
    for i = 1:lvl
        if v[:, i]' * refs[:, i] < 0
            refs[:, i] = -v[:, i]
        else
            refs[:, i] = v[:, i]
        end
    end
end

rotate_refs!(sys::ProjectedSystem, v, lvl) = rotate_refs!(sys.ref, v, lvl)

function push_params!(sys::ProjectedSystem, w, v, dH, interaction, d_inds)
    # update energies
    E = w / 2 / π
    push!(sys.ev, E)
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
    push!(sys.dθ, dθ)
    # update projected interaction operators
    # parenthsis ensure sparse matrix multiplication is performed first
    op = [sys.ref' * x * sys.ref for x in interaction]
    push!(sys.op, op)
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


function construct_projected_coupling(sys)
    proj_c = ProjectedCoupling(sys.s, sys.lvl)
    for (t_idx, op) in enumerate(sys.op)
        for j = 1:sys.lvl
            for i = (j + 1):(sys.lvl - j + 1)
                proj_c.a[t_idx, i, j] = sum((x) -> (x[i, i] - x[j, j])^2, op)
                proj_c.b[t_idx, i, j] = sum((x) -> abs2(x[i, j]), op)
                proj_c.c[t_idx, i, j] =
                    sum((x) -> x[i, j] * (x[i, i] - x[j, j]), op)
                proj_c.d[t_idx, i, j] =
                    sum((x) -> x[i, j] * (x[i, i] + x[j, j]), op)
            end
        end
    end
    proj_c
end


function construct_projected_TG(sys)
    proj_tg = ProjectedTG(sys.s, sys.lvl)
    for (idx, dθ) in enumerate(sys.dθ)
        proj_tg.G[idx, :] = dθ
        proj_tg.ω[idx, :] = sys.ev[idx]
    end
    fill!(proj_tg.T.mat, 0.0)
    proj_tg
end


macro unitary_landau_zener(θ)
    return quote
        local val = $(esc(θ))
        [cos(val) -sin(val); sin(val) cos(val)]
    end
end


function landau_zener_rotate_angle(sys::ProjectedSystem, rotation_point)
    non_rotation_idx = 1:(rotation_point - 1)
    rotation_idx = rotation_point:length(sys.s)
    dθ_itp = construct_interpolations(
        sys.s[rotation_idx],
        -get_dθ(sys, 2, 1)[rotation_idx],
        extrapolation="line",
    )
    θᴸ_2 = [
        quadgk(dθ_itp, sys.s[rotation_point], s)[1]
        for s in sys.s[rotation_idx]
    ]
    θᴸ_1 = zeros(rotation_point - 1)
    θᴸ = vcat(θᴸ_1, θᴸ_2)
end


function landau_zener_rotate(sys::ProjectedSystem, rotation_point)
    θ = landau_zener_rotate_angle(sys, rotation_point)
    non_rotation_idx = 1:(rotation_point - 1)
    rotation_idx = rotation_point:length(sys.s)
    proj_tg = ProjectedTG(sys.s, sys.lvl)
    proj_c = ProjectedCoupling(sys.s, sys.lvl)

    for idx in non_rotation_idx
        proj_tg.ω[idx, :] = sys.ev[idx]
        proj_tg.T[idx, :] = 0.0
        proj_tg.G[idx, :] = sys.dθ[idx]
        op = sys.op[idx]
        for j = 1:sys.lvl
            for i = (j + 1):(sys.lvl - j + 1)
                proj_c.a[idx, i, j] = sum((x) -> (x[i, i] - x[j, j])^2, op)
                proj_c.b[idx, i, j] = sum((x) -> abs2(x[i, j]), op)
                proj_c.c[idx, i, j] =
                    sum((x) -> x[i, j] * (x[i, i] - x[j, j]), op)
                proj_c.d[idx, i, j] =
                    sum((x) -> x[i, j] * (x[i, i] + x[j, j]), op)
            end
        end
    end

    for idx in rotation_idx
        U = @unitary_landau_zener(θ[idx])
        H = Diagonal(sys.ev[idx])
        H = U' * H * U
        proj_tg.ω[idx, :] = diag(H)
        proj_tg.G[idx, :] = 0.0
        op = [U' * x * U for x in sys.op[idx]]
        for j = 1:sys.lvl
            for i = (j + 1):(sys.lvl - j + 1)
                proj_c.a[idx, i, j] = sum((x) -> (x[i, i] - x[j, j])^2, op)
                proj_c.b[idx, i, j] = sum((x) -> abs2(x[i, j]), op)
                proj_c.c[idx, i, j] =
                    sum((x) -> x[i, j] * (x[i, i] - x[j, j]), op)
                proj_c.d[idx, i, j] =
                    sum((x) -> x[i, j] * (x[i, i] + x[j, j]), op)
                proj_tg.T[idx, i, j] = H[i, j]
            end
        end
    end
    proj_tg, proj_c
end


Base.summary(TG::ProjectedTG) = string(TYPE_COLOR, nameof(typeof(TG)), NO_COLOR)


function Base.show(io::IO, TG::ProjectedTG)
    print(io, summary(TG))
    print(io, " with size: ")
    show(io, (size(TG.ω, 2), size(TG.ω, 2)))
end


Base.summary(C::ProjectedCoupling) =
    string(TYPE_COLOR, nameof(typeof(C)), NO_COLOR)


function Base.show(io::IO, C::ProjectedCoupling)
    print(io, summary(C))
    print(io, " with size: ")
    show(io, (C.a.lvl, C.a.lvl))
end
