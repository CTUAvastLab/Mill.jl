struct Max{T} <: AggregationFunction
    C::T
end

Max(d::Int) = Max(param(zeros(Float32, d)))
Flux.@treelike Max

modelprint(io::IO, sm::Max{T}; pad=[]) where T = paddedprint(io, "Max{$(T)}($(length(sm.C)))")

# FORWARD PASS
(m::Max)(x, args...) = segmented_max(x, m.c, args...)
(m::Max)(x::ArrayNode, args...) = mapdata(x -> m(x, m.c, args...), x)
(m::Max{<:TrackedVector})(x::ArrayNode, args...) = mapdata(x -> n(x, m.c, args...), x)

function segmented_max(x::Matrix, C::Vector, bags::AbstractBags)
    o = similar(x, size(x,1), length(bags))
    fill!(o, typemin(eltype(x)))
    for (j, b) in enumerate(bags)
        if isempty(b)
            for i in 1:size(x, 1)
                @inbounds o[i, j] = C[i]
            end
        else
            for bi in b
                for i in 1:size(x, 1)
                    @inbounds o[i, j] = max(o[i, j], x[i, bi])
                end
            end
        end
    end
    o
end

function segmented_max(x::Matrix, C::Vector, bags::AbstractBags, mask::Vector{Bool})
    o = similar(x, size(x,1), length(bags))
    fill!(o, typemin(eltype(x)))
    for (j, b) in enumerate(bags)
        if isempty(b)
            for i in 1:size(x, 1)
                @inbounds o[i, j] = C[i]
            end
        else
            for bi in b
                @inbounds !mask[bi] && continue
                for i in 1:size(x, 1)
                    @inbounds o[i, j] = max(o[i, j], x[i, bi])
                end
            end
        end
    end
    o
end

# weighted version (identical to non-weighted)
segmented_max(x::Matrix, bags::AbstractBags, C::Vector, w::Vector) = segmented_max(x, C, bags)

# BACKWARD
(m::Max{<:AbstractVector})(x::TrackedMatrix, args...) = _max_grad(x, m.C, args...)
(m::Max{<:TrackedVector})(x, args...) = _max_grad(x, m.C, args...)
(m::Max{<:TrackedVector})(x::TrackedMatrix, args...) = _max_grad(x, m.C, args...)

_max_grad(x, C, args...) = Flux.Tracker.track(_max_grad, x, C, args...)
Flux.Tracker.@grad function _max_grad(x, C, args...)
    n = segmented_max(Flux.data(x), Flux.data(C), Flux.data.(args)...)
    grad = Δ -> segmented_max_back(Δ, x, C, args..., n)
    n, grad
end

function segmented_max_back(Δ, x::TrackedMatrix, C::TrackedVector, bags::AbstractBags)
    Δ = Flux.data(Δ)
    x = Flux.data(x)
    C = Flux.data(C)
    dx = zero(x)
    dc = zero(C)
    v = similar(x, size(x,1))
    idxs = zeros(Int, size(x,1))
    for (j, b) in enumerate(bags)
        if isempty(b)
            for i in 1:size(x, 1)
                @inbounds dc[i] += Δ[i, j]
            end
        else
            fill!(v, typemin(eltype(x)))
            for bi in b
                for i in 1:size(x, 1)
                    if v[i] < x[i, bi]
                        @inbounds idxs[i] = bi
                        @inbounds v[i] = x[i, bi]
                    end
                end
            end
            for i in 1:size(x, 1)
                @inbounds dx[i, idxs[i]] = Δ[i, j]
            end
        end
    end
    dx, dc, nothing
end

function segmented_max_back(Δ, x::Matrix, C::TrackedVector, bags::AbstractBags)
    Δ = Flux.data(Δ)
    C = Flux.data(C)
    dc = zero(C)
    for (j, b) in enumerate(bags)
        if isempty(b)
            for i in 1:size(x, 1)
                @inbounds dc[i] += Δ[i, j]
            end
        end
    end
    nothing, dc, nothing
end

function segmented_max_back(Δ, x::TrackedMatrix, C::Vector, bags::AbstractBags)
    Δ = Flux.data(Δ)
    x = Flux.data(x)
    dx = zero(x)
    v = similar(x, size(x,1))
    idxs = zeros(Int, size(x,1))
    for (j, b) in enumerate(bags)
        if !isempty(b)
            fill!(v, typemin(eltype(x)))
            for bi in b
                for i in 1:size(x, 1)
                    if v[i] < x[i, bi]
                        @inbounds idxs[i] = bi
                        @inbounds v[i] = x[i, bi]
                    end
                end
            end
            for i in 1:size(x, 1)
                @inbounds dx[i, idxs[i]] = Δ[i, j]
            end
        end
    end
    dx, nothing, nothing
end

segmented_max_back(Δ, x::Matrix, C::Vector, bags::AbstractBags) = (nothing, nothing, nothing)

# weighted versions
function segmented_max_back(Δ, x::Union{Matrix, TrackedMatrix}, C::Union{Vector, TrackedVector}, bags::AbstractBags, w::TrackedVector)
    tuple(segmented_max_back(Δ, x, C, bags)..., zero(w))
end
function segmented_max_back(Δ, x::Union{Matrix, TrackedMatrix}, C::Union{Vector, TrackedVector}, bags::AbstractBags, w::Vector)
    tuple(segmented_max_back(Δ, x, C, bags)..., nothing)
end
