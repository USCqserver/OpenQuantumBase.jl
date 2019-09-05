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
    s
    """``(σ_m - σ_n)^2``"""
    a
    """``|σ_{mn}|^2``"""
    b
    """``σ_{mn}(σ_m-σ_n)``"""
    c
    """``σ_{mn}(σ_m+σ_n)``"""
    d
end


function ProjectedSystem(s, size, lvl)
    len = length(s)
    ev = Vector{Vector{Float64}}()
    dθ = Vector{Vector{Float64}}()
    op = Vector{Vector{Matrix{Float64}}}()
    ProjectedSystem(s, ev, dθ, op, zeros(size, lvl), lvl)
end


"""
    project_to_lowlevel(H::AbstractHamiltonian{T}, dH, coupling, s_axis; ref=zeros((0,2)), tol=1e-4, lvl=2)

Project a Hamiltonian `H` to the lowest `lvl` level subspace. `dH` is the derivative of Hamiltonian and `coupling` is the system-bath interaction operator. They should be callable with a single argument -- annealing parameter `s`. `s_axis` is the (unitless) times to calculate the projection.
"""
function project_to_lowlevel(
    H::AbstractHamiltonian{T},
    dH,
    coupling,
    s_axis::AbstractArray{T,1};
    ref = zeros((0, 2)), tol = 1e-4, lvl = 2
) where T <: Real
    projected_system = ProjectedSystem(s_axis, H.size[1], lvl)
    for s in s_axis
        w, v = eigen_decomp(H, s; level = lvl, tol = tol, v0 = ref[:, 1])
        push_params!(projected_system, w, v, dH(s), coupling(s))
        ref = projected_system.ref
    end
    projected_system
end


function project_to_lowlevel(
    H::AbstractHamiltonian{T},
    dH,
    coupling,
    s_axis;
    ref = zeros((0, 2)), tol = 1e-4, lvl = 2
) where T <: Complex
    @warn "The projection method only works with real Hamitonians. Convert the complex Hamiltonian to real one."
    H_real = to_real(H)
    project_to_lowlevel(
        H_real,
        dH,
        coupling,
        s_axis,
        ref = ref,
        tol = tol,
        lvl = lvl
    )
end


function push_params!(sys::ProjectedSystem, w, v, dH, interaction)
    push!(sys.ev, w)
    # update reference vectors
    for i = 1:sys.lvl
        if v[:, i]' * sys.ref[:, i] < 0
            sys.ref[:, i] = -v[:, i]
        else
            sys.ref[:, i] = v[:, i]
        end
    end
    # update dθ
    dθ = Vector{Float64}()
    for j = 1:sys.lvl
        for i = (j+1):sys.lvl
            vi = @view sys.ref[:, i]
            vj = @view sys.ref[:, j]
            t = vi' * dH * vj / (w[j] - w[i])
            push!(dθ, t)
        end
    end
    push!(sys.dθ, dθ)
    # update projected interaction operators
    # parenthsis ensure sparse matrix multiplication is performed first
    op = [sys.ref' * (x * sys.ref) for x in interaction]
    push!(sys.op, op)
end


"""
    get_dθ(sys::ProjectedSystem, i=1, j=2)

Get the geometric terms between i, j energy levels from `ProjectedSystem`.
"""
function get_dθ(sys, i = 1, j = 2)
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
    t_dim = length(sys.s)
    s_dim = sys.lvl*(sys.lvl-1) ÷ 2
    a = Matrix{Float64}(undef, t_dim, s_dim)
    b = Matrix{Float64}(undef, t_dim, s_dim)
    c = Matrix{Float64}(undef, t_dim, s_dim)
    d = Matrix{Float64}(undef, t_dim, s_dim)
    for (t_idx, op) in enumerate(sys.op)
        at = Vector{Float64}()
        bt = Vector{Float64}()
        ct = Vector{Float64}()
        dt = Vector{Float64}()
        for j = 1:sys.lvl
            for i = (j+1):(sys.lvl-j+1)
                s_idx = linear_idx_off(i, j, sys.lvl)
                a[t_idx, s_idx] = sum((x) -> (x[i, i] - x[j, j])^2, op)
                b[t_idx, s_idx] = sum((x) -> abs2(x[i, j]), op)
                c[t_idx, s_idx] = sum((x) -> x[i, j] * (x[i, i] - x[j, j]), op)
                d[t_idx, s_idx] = sum((x) -> x[i, j] * (x[i, i] + x[j, j]), op)
            end
        end
    end
    ProjectedCoupling(s, a, b, c, d)
end
