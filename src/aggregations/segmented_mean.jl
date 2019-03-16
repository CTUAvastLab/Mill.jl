struct Mean{T} <: AggregationFunction
    C::T
end

Mean(d::Int) = Mean(param(zeros(Float32, d)))
Flux.@treelike Mean

modelprint(io::IO, sm::Mean{T}; pad=[]) where T = paddedprint(io, "Mean{$(T)}($(length(sm.C)))")
# FORWARD PASS
(m::Mean)(x, args...) = segmented_mean(x, m.c, args...)
(m::Mean)(x::ArrayNode, args...) = mapdata(x -> m(x, m.c, args...), x)
(m::Mean{<:TrackedVector})(x::ArrayNode, args...) = mapdata(x -> n(x, m.c, args...), x)

function segmented_mean(x::Matrix, c::Vector, bags::AbstractBags)
    o = zeros(eltype(x), size(x, 1), length(bags))
    for (j, b) in enumerate(bags)
        if isempty(b)
            for i in 1:size(x, 1)
                @inbounds o[i, j] = C[i]
            end
        else
            for bi in b
                for i in 1:size(x, 1)
                    @inbounds o[i,j] += x[i,bi] / length(b)
                end
            end
        end
    end
    o
end

function segmented_mean(x::Matrix, c::Vector, bags::AbstractBags, w::Vector)
    o = zeros(eltype(x), size(x, 1), length(bags))
    for (j, b) in enumerate(bags)
        if isempty(b)
            for i in 1:size(x, 1)
                @inbounds o[i, j] = C[i]
            end
        else
            ws = sum(@view w[b])
            for bi in b
                for i in 1:size(x, 1)
                    @inbounds o[i,j] += w[bi] * x[i,bi] / ws
                end
            end
        end
    end
    o
end

function segmented_mean(x::Matrix, c::Vector, bags::AbstractBags, mask::Vector{Bool})
    o = zeros(eltype(x), size(x, 1), length(bags))
    for (j, b) in enumerate(bags)
        if isempty(b)
            for i in 1:size(x, 1)
                @inbounds o[i, j] = C[i]
            end
        else
            for bi in b
                @inbounds !mask[bi] && continue
                for i in 1:size(x, 1)
                    @inbounds o[i,j] += x[i,bi] / length(b)
                end
            end
        end
    end
    o
end

function segmented_mean(x::Matrix, c::Vector, bags::AbstractBags, w::Vector, mask::Vector{Bool})
    o = zeros(eltype(x), size(x, 1), length(bags))
    @inbounds for (j, b) in enumerate(bags)
        ws = sum(@view w[b])
        for bi in b
            for i in 1:size(x, 1)
                o[i,j] += w[bi] * x[i,bi] / ws
            end
        end
    end
    o
end

function segmented_mean_back(Δ, x::TrackedMatrix, bags::AbstractBags)
    x = Flux.data(x)
    Δ = Flux.data(Δ)
    dx = similar(x)
    @inbounds for (j, b) in enumerate(bags)
        for bi in b
            for i in 1:size(x,1)
                dx[i,bi] = Δ[i,j] / length(b)
            end
        end
    end
    dx, nothing
end

function segmented_mean_back(Δ, x::TrackedMatrix, bags::AbstractBags, w::Vector)
    x = Flux.data(x)
    Δ = Flux.data(Δ)
    dx = similar(x)
    @inbounds for (j, b) in enumerate(bags)
        ws = sum(@view w[b])
        for bi in b
            for i in 1:size(x,1)
                dx[i, bi] = w[bi] * Δ[i, j] / ws
            end
        end
    end
    dx, nothing, nothing
end

function segmented_mean_back(Δ, x::Matrix, bags::AbstractBags, w::TrackedVector)
    x = Flux.data(x)
    Δ = Flux.data(Δ)
    w = Flux.data(w)
    n = segmented_mean(x, bags, w)
    dw = zero(w)
    @inbounds for (j, b) in enumerate(bags)
        ws = sum(@view w[b])
        for bi in b
            for i in 1:size(x,1)
                dw[bi] += Δ[i, j] * (x[i, bi] - n[i, j]) / ws
            end
        end
    end
    nothing, nothing, dw
end

function segmented_mean_back(Δ, x::TrackedMatrix, bags::AbstractBags, w::TrackedVector)
    x = Flux.data(x)
    Δ = Flux.data(Δ)
    w = Flux.data(w)
    n = segmented_mean(x, bags, w)
    dx = similar(x)
    @inbounds for (j, b) in enumerate(bags)
        ws = sum(@view w[b])
        for bi in b
            for i in 1:size(x,1)
                dx[i, bi] = w[bi] * Δ[i, j] / ws
            end
        end
    end
    dw = zero(w)
    @inbounds for (j, b) in enumerate(bags)
        ws = sum(@view w[b])
        for bi in b
            for i in 1:size(x,1)
                dw[bi] += Δ[i, j] * (x[i, bi] - n[i, j]) / ws
            end
        end
    end
    dx, nothing, dw
end

