struct SegmentedSum{T} <: AggregationFunction
    C::T
end

Flux.@functor SegmentedSum

SegmentedSum(d::Int) = SegmentedSum(zeros(Float32, d))

Base.show(io::IO, sm::SegmentedSum) = print(io, "SegmentedSum($(length(sm.C)))\n")
modelprint(io::IO, sm::SegmentedSum; pad=[]) = paddedprint(io, "SegmentedSum($(length(sm.C)))")

(m::SegmentedSum)(x::MaybeMatrix, bags::AbstractBags, w=nothing) = segmented_sum_forw(x, m.C, bags, w)
function (m::SegmentedSum)(x::AbstractMatrix, bags::AbstractBags, w::AggregationWeights, mask::AbstractVector)
    segmented_sum_forw(x .* mask', m.C, bags, w)
end

segmented_sum_forw(::Missing, C::AbstractVector, bags::AbstractBags, w) = repeat(C, 1, length(bags))
function segmented_sum_forw(x::AbstractMatrix, C::AbstractVector, bags::AbstractBags, w::AggregationWeights) 
    y = zeros(eltype(x), size(x, 1), length(bags))
    @inbounds for (bi, b) in enumerate(bags)
        if isempty(b)
            for i in eachindex(C)
                y[i, bi] = C[i]
            end
        else
            for j in b
                for i in 1:size(x, 1)
                    y[i, bi] += weight(w, i, j) * x[i, j]
                end
            end
        end
    end
    y
end

function segmented_sum_back(Δ, y, x, C, bags, w) 
    dx = similar(x)
    dC = zero(C)
    dw = isnothing(w) ? nothing : zero(w)
    @inbounds for (bi, b) in enumerate(bags)
        if isempty(b)
            for i in eachindex(C)
                dC[i] += Δ[i, bi]
            end
        else
            for j in b
                for i in 1:size(x, 1)
                    dx[i, j] = weight(w, i, j) * Δ[i, bi]
                    ∇dw_segmented_sum!(dw, Δ, x, y, w, i, j, bi)
                end
            end
        end
    end
    dx, dC, nothing, dw
end

function segmented_sum_back(Δ, y, x::Missing, C, bags, w::Nothing) 
    dC = zero(C)
    @inbounds for (bi, b) in enumerate(bags)
        for i in eachindex(C)
            dC[i] += Δ[i, bi]
        end
    end
    nothing, dC, nothing, nothing
end

∇dw_segmented_sum!(dw::Nothing, Δ, x, y, w::Nothing, i, j, bi) = nothing
function ∇dw_segmented_sum!(dw::AbstractVector, Δ, x, y, w::AbstractVector, i, j, bi) 
    dw[j] += Δ[i, bi] * (x[i, j])
end
function ∇dw_segmented_sum!(dw::AbstractMatrix, Δ, x, y, w::AbstractMatrix, i, j, bi)
    dw[i, j] += Δ[i, bi] * (x[i, j])
end

@adjoint function segmented_sum_forw(args...)
    y = segmented_sum_forw(args...)
    grad = Δ -> segmented_sum_back(Δ, y, args...)
    y, grad
end
