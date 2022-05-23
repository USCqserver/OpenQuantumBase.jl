"""
$(TYPEDEF)

`GapIndices` contains unique gaps and the corresponding indices. The information is used to calculate the Davies generator.

# Fields

$(FIELDS)
"""
struct GapIndices
    "Unique positive gaps"
    uniq_w
    "a indices for the corresponding gaps in uniq_w"
    uniq_a
    "b indices for the corresponding gaps in uniq_w"
    uniq_b
    "a indices for the 0 gap"
    a0
    "b indices for the 0 gap"
    b0
end

function GapIndices(w::Vector{T}; digits::Integer=8, sigdigits::Integer=8) where T<:Real
    l = length(w)
    gaps = Float64[]
    a_idx = Vector{Int}[]
    b_idx = Vector{Int}[]
    a0_idx = Int[]
	b0_idx = Int[]
    for i in 1:l-1
        for j in i+1:l
            gap = w[j] - w[i]
            if abs(gap) ≤ 10^(-digits)
                push!(a0_idx, i)
                push!(b0_idx, j)
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
    GapIndices(gaps, a_idx, b_idx, a0_idx, b0_idx)
end