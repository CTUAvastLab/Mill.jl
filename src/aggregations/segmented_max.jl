struct SegmentedMax{T} <: AggregationFunction
    C::T
end

Flux.@functor SegmentedMax

SegmentedMax(d::Int) = SegmentedMax(zeros(Float32, d))

Base.show(io::IO, sm::SegmentedMax) = print(io, "SegmentedMax($(length(sm.C)))\n")
modelprint(io::IO, sm::SegmentedMax; pad=[]) = paddedprint(io, "SegmentedMax($(length(sm.C)))")

(m::SegmentedMax)(x::MaybeMatrix, bags::AbstractBags, w=nothing) = segmented_max_forw(x, m.C, bags)
function (m::SegmentedMax)(x::AbstractMatrix, bags::AbstractBags, w::AggregationWeights, mask::AbstractVector)
    segmented_max_forw(x .+ typemin(T) * mask', m.C, bags)
end

segmented_max_forw(::Missing, C::AbstractVector, bags::AbstractBags) = repeat(C, 1, length(bags))
function segmented_max_forw(x::AbstractMatrix, C::AbstractVector, bags::AbstractBags) 
    y = fill(typemin(eltype(x)), size(x, 1), length(bags))
    @inbounds for (bi, b) in enumerate(bags)
        if isempty(b)
            for i in eachindex(C)
                y[i, bi] = C[i]
            end
        else
            for j in b
                for i in 1:size(x, 1)
                    y[i, bi] = max(y[i, bi], x[i, j])
                end
            end
        end
    end
    y
end

function segmented_max_back(Δ, y, x, C, bags) 
    dx = zero(x)
    dC = zero(C)
    v = similar(x, size(x, 1))
    idxs = zeros(Int, size(x, 1))
    @inbounds for (bi, b) in enumerate(bags)
        if isempty(b)
            for i in eachindex(C)
                dC[i] += Δ[i, bi]
            end
        else
            fi = first(b)
            v .= x[:,fi]
            idxs .= fi
            for j in b
                for i in 1:size(x,1)
                    if v[i] < x[i, j]
                        idxs[i] = j
                        v[i] = x[i, j]
                    end
                end
            end
            for i in 1:size(x, 1)
                dx[i, idxs[i]] = Δ[i, bi]
            end
        end
    end
    dx, dC, nothing, nothing
end

function segmented_max_back(Δ, y, x::Missing, C, bags) 
    dC = zero(C)
    @inbounds for (bi, b) in enumerate(bags)
        for i in eachindex(C)
            dC[i] += Δ[i, bi]
        end
    end
    nothing, dC, nothing, nothing
end

@adjoint function segmented_max_forw(args...)
    y = segmented_max_forw(args...)
    grad = Δ -> segmented_max_back(Δ, y, args...)
    y, grad
end
