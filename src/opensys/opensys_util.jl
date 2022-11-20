"""
$(TYPEDEF)

`GapIndices` contains unique gaps and the corresponding indices. The information is used to calculate the Davies generator.

# Fields

$(FIELDS)
"""
struct GapIndices
    "Energies"
    w::AbstractVector{Real}
    "Unique positive gaps"
    uniq_w::Vector{Real}
    "a indices for the corresponding gaps in uniq_w"
    uniq_a
    "b indices for the corresponding gaps in uniq_w"
    uniq_b
    "a indices for the 0 gap"
    a0::Vector{Int}
    "b indices for the 0 gap"
    b0::Vector{Int}
end

function GapIndices(w::AbstractVector{T}, digits::Integer, sigdigits::Integer) where T<:Real
    l = length(w)
    gaps = Float64[]
    a_idx = Vector{Int}[]
    b_idx = Vector{Int}[]
    a0_idx = Int[]
	b0_idx = Int[]
    for i in 1:l-1
        for j in i+1:l
            gap = w[j] - w[i]
            if abs(gap) ≤ 10.0^(-digits)
                push!(a0_idx, i)
                push!(b0_idx, j)
                push!(a0_idx, j)
                push!(b0_idx, i)
            else
                gap = round(gap, sigdigits=sigdigits)
                idx = searchsortedfirst(gaps, gap)
                if idx == length(gaps) + 1
                    push!(gaps, gap)
					push!(a_idx, [i])
					push!(b_idx, [j])
                elseif gaps[idx] == gap
                    push!(a_idx[idx], i)
                    push!(b_idx[idx], j)
                else
                    insert!(gaps, idx, gap)
                    insert!(a_idx, idx, [i])
                    insert!(b_idx, idx, [j])
                end
            end
        end
    end
    append!(a0_idx, 1:l)
	append!(b0_idx, 1:l)
    GapIndices(w, gaps, a_idx, b_idx, a0_idx, b0_idx)
end

positive_gap_indices(G::GapIndices) = zip(G.uniq_w, G.uniq_a, G.uniq_b)
zero_gap_indices(G::GapIndices) = G.a0, G.b0
gap_matrix(G::GapIndices) = G.w' .- G.w
get_lvl(G::GapIndices) = length(G.w)
get_gaps_num(G::GapIndices) = 2*length(G.uniq_w)+1

# TODO: merge `build_correlation` function with `GapIndices`
"""
$(SIGNATURES)

Build `GapIndices` type from a list of energies.

...
# Arguments
- `w::AbstractVector`: energies of the Hamiltonian.
- `digits::Integer`: the number of digits to keep when checking if a gap is zero.
- `sigdigits::Interger`: the number of significant digits when rounding non-zero gaps for comparison.
- `cutoff_freq::Real`: gaps that are larger than the cutoff frequency (in the 2π frequency unit) are neglected.
- `truncate_lvl::Integer`: energy levels that are higher than the `truncate_lvl` are neglected.
...
"""
function build_gap_indices(w::AbstractVector, digits::Integer, sigdigits::Integer, cutoff_freq::Real, truncate_lvl::Integer)
    l = truncate_lvl
    gaps = Float64[]
    a_idx = Vector{Int}[]
    b_idx = Vector{Int}[]
    a0_idx = Int[]
	b0_idx = Int[]
    for i in 1:l-1
        for j in i+1:l
            gap = w[j] - w[i]
            if abs(gap) ≤ 10.0^(-digits)
                push!(a0_idx, i)
                push!(b0_idx, j)
                push!(a0_idx, j)
                push!(b0_idx, i)
            elseif abs(gap) ≤ cutoff_freq
                gap = round(gap, sigdigits=sigdigits)
                idx = searchsortedfirst(gaps, gap)
                if idx == length(gaps) + 1
                    push!(gaps, gap)
					push!(a_idx, [i])
					push!(b_idx, [j])
                elseif gaps[idx] == gap
                    push!(a_idx[idx], i)
                    push!(b_idx[idx], j)
                else
                    insert!(gaps, idx, gap)
                    insert!(a_idx, idx, [i])
                    insert!(b_idx, idx, [j])
                end
            end
        end
    end
    append!(a0_idx, 1:l)
	append!(b0_idx, 1:l)
    GapIndices(w, gaps, a_idx, b_idx, a0_idx, b0_idx)
end