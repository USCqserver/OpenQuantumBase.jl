"""
$(TYPEDEF)

Defines a total Liouvillian to feed to the solver using the `DiffEqOperator` interface. It contains both closed-system and open-system Liouvillians.

# Fields

$(FIELDS)
"""
struct DiffEqLiouvillian{diagonalization,adiabatic_frame}
    "Hamiltonian"
    H::AbstractHamiltonian
    "Open system in eigenbasis"
    opensys_eig::Vector{AbstractLiouvillian}
    "Open system in normal basis"
    opensys::Vector{AbstractLiouvillian}
    "Levels to truncate"
    lvl::Int
    "Internal cache"
    u_cache::Any
end

"""
$(SIGNATURES)

The constructor of the `DiffEqLiouvillian` type. `opensys_eig` is a list of open-system Liouvillians that which require diagonalization of the Hamiltonian. `opensys` is a list of open-system Liouvillians which does not require diagonalization of the Hamiltonian. `lvl` is the truncation levels of the energy eigenbasis if the method supports the truncation.
"""
function DiffEqLiouvillian(H, opensys_eig, opensys, lvl)
    # for DenseHamiltonian smaller than 10×10, do not truncate
    if !(typeof(H) <: AbstractSparseHamiltonian) && (size(H, 1) <= 10)
        lvl = size(H, 1)
        u_cache = similar(get_cache(H))
    else
        # for SparseHamiltonian, we will create dense matrix cache
        # for the truncated subspace
        u_cache = Matrix{eltype(H)}(undef, lvl, lvl)
    end
    diagonalization = isempty(opensys_eig) ? false : true
    adiabatic_frame = typeof(H) <: AdiabaticFrameHamiltonian
    DiffEqLiouvillian{diagonalization,adiabatic_frame}(H, opensys_eig, opensys, lvl, u_cache)
end

function (Op::DiffEqLiouvillian{false,false})(du, u, p, t)
    s = p(t)
    Op.H(du, u, p, s)
    for lv in Op.opensys
        lv(du, u, p, t)
    end
end

function update_cache!(cache, Op::DiffEqLiouvillian{false,false}, p, t)
    update_cache!(cache, Op.H, p, p(t))
    for lv in Op.opensys
        update_cache!(cache, lv, p, t)
    end
end

function update_vectorized_cache!(cache, Op::DiffEqLiouvillian{false,false}, p, t)
    update_vectorized_cache!(cache, Op.H, p, p(t))
    for lv in Op.opensys
        update_vectorized_cache!(cache, lv, p, t)
    end
end

function (Op::DiffEqLiouvillian{true,false})(du, u, p, t)
    s = p(t)
    w, v = Op.H.EIGS(Op.H, s, Op.lvl)
    ω_ba = transpose(w) .- w
    ρ = v' * u * v
    H = Diagonal(w)
    Op.u_cache .= -1.0im * (H * ρ - ρ * H)
    for lv in Op.opensys_eig
        lv(Op.u_cache, ρ, ω_ba, v, s)
    end
    mul!(du, v, Op.u_cache * v')
    for lv in Op.opensys
        lv(du, u, p, t)
    end
end

function update_cache!(cache, Op::DiffEqLiouvillian{true,false}, p, t::Real)
    s = p(t)
    w, v = Op.H.EIGS(Op.H, s, Op.lvl)
    ω_ba = transpose(w) .- w
    # initialze the cache as Hamiltonian in eigenbasis
    fill!(Op.u_cache, 0.0)
    for i = 1:length(w)
        @inbounds Op.u_cache[i, i] = -1.0im * w[i]
    end
    for lv in Op.opensys_eig
        update_cache!(Op.u_cache, lv, ω_ba, v, s)
    end
    mul!(cache, v, Op.u_cache * v')
    for lv in Op.opensys
        update_cache!(cache, lv, p, t)
    end
end

function (Op::DiffEqLiouvillian{true,true})(du, u, p, t)
    s = p(t)
    H = Op.H(p.tf, s)
    w = diag(H)
    ω_ba = transpose(w) .- w
    du .= -1.0im * (H * u - u * H)
    for lv in Op.opensys_eig
        lv(du, u, ω_ba, s)
    end
end

# """
# $(TYPEDEF)

# Defines an open system DiffEqOperator in the multiple eigenbasis.

# # Fields

# $(FIELDS)
# """
# struct OpenSysOpHybrid{adiabatic_frame}
#     """Hamiltonian"""
#     H::AbstractHamiltonian
#     """Open system in energy eigenbasis"""
#     opensys_eig::Vector{AbstractLiouvillian}
#     """Open system in normal basis"""
#     opensys::Vector{AbstractLiouvillian}
#     """Levels to truncate"""
#     lvl::Int
#     """Internal cache"""
#     u_cache::Any
# end

# function OpenSysOpHybrid(H, open_eig, op, lvl)
#     # for DenseHamiltonian smaller than 10×10, do not truncate
#     if !(typeof(H) <: AbstractSparseHamiltonian) && (size(H, 1) <= 10)
#         lvl = size(H, 1)
#         u_cache = similar(get_cache(H))
#     else
#         # for SparseHamiltonian, we will create dense matrix cache
#         # for the truncated subspace
#         u_cache = Matrix{eltype(H)}(undef, lvl, lvl)
#     end
#     adiabatic_frame = typeof(H) <: AdiabaticFrameHamiltonian
#     OpenSysOpHybrid{adiabatic_frame}(H, open_eig, op, lvl, u_cache)
# end

# function update_cache!(cache, Op::OpenSysOpHybrid{false}, p, t::Real)
#     s = p(t)
#     w, v = Op.H.EIGS(Op.H, s, Op.lvl)
#     ω_ba = transpose(w) .- w
#     # initialze the cache as Hamiltonian in eigenbasis
#     fill!(Op.u_cache, 0.0)
#     for i = 1:length(w)
#         @inbounds Op.u_cache[i, i] = -1.0im * w[i]
#     end
#     for lv in Op.opensys_eig
#         update_cache!(Op.u_cache, lv, ω_ba, v, s)
#     end
#     mul!(cache, v, Op.u_cache * v')
#     for lv in Op.opensys
#         update_cache!(cache, lv, p, t)
#     end
# end
