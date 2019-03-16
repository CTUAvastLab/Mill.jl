# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4908336/
struct LSE{T}
    p::T
end

LSE(d::Int) = LSE(param(randn(Float32, d)))
Flux.@treelike LSE

Base.show(io::IO, n::LSE) = print(io, "LogSumExp($(length(n.p)))")

function segmented_lse(x::Matrix, p::Vector, bags::AbstractBags)
    o = zeros(eltype(x), size(x, 1), length(bags))
    @inbounds for (j, b) in enumerate(bags)
        for bi in b
            for i in 1:size(x, 1)
                o[i, j] += exp(p[i] * x[i, bi])
            end
        end
        o[:, j] .= (log.(o[:, j]) .- log(length(b)))
    end
    o ./ p
end

segmented_lse(x::Matrix, p::Vector, bags::AbstractBags, w::Vector) = segmented_lse(x, p, bags)

function segmented_lse_back(Δ, x::TrackedMatrix, p::Vector, bags::AbstractBags, n::Matrix)
    x = Flux.data(x)
    Δ = Flux.data(Δ)
    dx = zero(x)
    ss = zero(p)
    @inbounds for (j, b) in enumerate(bags)
        ss .= 0
        for bi in b
            for i in 1:size(x,1)
                e = exp(p[i] * x[i, bi])
                dx[i, bi] = Δ[i, j] * e
                ss[i] += e
            end
        end
        dx[:, b] ./= ss
    end
    dx, nothing, nothing
end

segmented_lse_back(Δ, x::TrackedMatrix, p::Vector, bags::AbstractBags, w::Vector, n::Matrix) = tuple(segmented_lse_back(Δ, x, p, bags, n)..., nothing)

function segmented_lse_back(Δ, x::TrackedMatrix, p::TrackedVector, bags::AbstractBags, n::Matrix)
    x = Flux.data(x)
    p = Flux.data(p)
    Δ = Flux.data(Δ)
    dx = zero(x)
    dp = zero(p)
    ss1 = zero(p)
    ss2 = zero(p)
    @inbounds for (j, b) in enumerate(bags)
        ss1 .= ss2 .= 0
        for bi in b
            for i in 1:size(x,1)
                e = exp(p[i] * x[i, bi])
                dx[i, bi] = Δ[i, j] * e
                ss1[i] += e
                ss2[i] += x[i, bi] * e
            end
        end
        dx[:, b] ./= ss1
        dp .+= Δ[:, j] .* (ss2 ./ ss1 - n[:, j])
    end
    dx, dp ./ p, nothing
end

segmented_lse_back(Δ, x::TrackedMatrix, p::TrackedVector, bags::AbstractBags, w::Vector, n::Matrix) = tuple(segmented_lse_back(Δ, x, p, bags, n)..., nothing)

function segmented_lse_back(Δ, x::Matrix, p::TrackedVector, bags::AbstractBags, n::Matrix)
    p = Flux.data(p)
    Δ = Flux.data(Δ)
    dp = zero(p)
    ss1 = zero(p)
    ss2 = zero(p)
    @inbounds for (j, b) in enumerate(bags)
        ss1 .= ss2 .= 0
        for bi in b
            for i in 1:size(x,1)
                e = exp(p[i] * x[i, bi])
                ss1[i] += e
                ss2[i] += x[i, bi] * e
            end
        end
        dp .+= Δ[:, j] .* (ss2 ./ ss1 - n[:, j])
    end
    nothing, dp ./ p, nothing, nothing
end

segmented_lse_back(Δ, x::Matrix, p::TrackedVector, bags::AbstractBags, w::Vector, n::Matrix) = tuple(segmented_lse_back(Δ, x, p, bags, n)..., nothing)

(n::LSE)(x, args...) = let m = maximum(x, dims=2)
    m .+ segmented_lse(x .- m, n.p, args...)
end
(n::LSE)(x::ArrayNode, args...) = mapdata(x -> n(x, args...), x)

(n::LSE{<:TrackedVector})(x::ArrayNode, args...) = mapdata(x -> n(x, args...), x)

# both x and p can be params
(n::LSE{<:AbstractVector})(x::TrackedMatrix, args...) = _lse_grad(x, n.p, args...)
(n::LSE{<:TrackedVector})(x, args...) = _lse_grad(x, n.p, args...)
(n::LSE{<:TrackedVector})(x::TrackedMatrix, args...) = _lse_grad(x, n.p, args...)

_lse_grad(x, p, args...) = let m = maximum(x, dims=2)
    m .+ Flux.Tracker.track(_lse_grad, x .- m, p, args...)
end

Flux.Tracker.@grad function _lse_grad(x, p, args...)
    n = segmented_lse(Flux.data(x), Flux.data(p), Flux.data.(args)...)
    grad = Δ -> segmented_lse_back(Δ, x, p, args..., n)
    n, grad
end
