struct SegmentedMax{T} <: AggregationFunction
    C::T
end

Flux.@treelike SegmentedMax

SegmentedMax(d::Int) = SegmentedMax(zeros(Float32, d))

Base.show(io::IO, sm::SegmentedMax) = print(io, "SegmentedMax($(length(sm.C)))\n")
modelprint(io::IO, sm::SegmentedMax; pad=[]) = paddedprint(io, "SegmentedMax($(length(sm.C)))")

(m::SegmentedMax)(x::ArrayNode, args...) = mapdata(x -> m(x, args...), x)
(m::SegmentedMax)(x, args...) = segmented_max(x, m.C, args...)

segmented_max(x, C, bags) = segmented_max(x, C, bags, nothing)

function segmented_max(x::Missing, C::AbstractVector, bags::AbstractBags, w, mask=nothing)
    repeat(C, 1, length(bags))
end

function segmented_max(x::AbstractMatrix, C::AbstractVector, bags::AbstractBags, w::AggregationWeights, mask::AbstractVector)
    segmented_max(x .* mask', C, bags, w, nothing)
end

function segmented_max(x::AbstractMatrix, C::AbstractVector, bags::AbstractBags, w::AggregationWeights) 
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

function segmented_max_back(Δ, y, x, C, bags, w=nothing) 
    dx = zero(x)
    dC = zero(C)
    dw = (w == nothing) ? nothing : zero(w)
    v = similar(x, size(x, 1))
    idxs = zeros(Int, size(x, 1))
    @inbounds for (bi, b) in enumerate(bags)
        fill!(v, typemin(eltype(x)))
        if isempty(b)
            for i in eachindex(C)
                dC[i] += Δ[i, bi]
            end
        else
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
    (dx, dC, nothing, dw)
end

Zygote.@adjoint function segmented_max(args...)
    y = segmented_max(args...)
    grad = Δ -> segmented_max_back(Δ, y, args...)
    y, grad
end
