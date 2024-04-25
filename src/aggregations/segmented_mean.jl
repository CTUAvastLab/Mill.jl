"""
    SegmentedMean{V <: AbstractVector{<:Number}}} <: AbstractAggregation

[`AbstractAggregation`](@ref) implementing segmented mean aggregation:

``
f(\\{x_1, \\ldots, x_k\\}) = \\frac{1}{k} \\sum_{i = 1}^{k} x_i
``

Stores a vector of parameters `ψ` that are filled into the resulting matrix in case an empty bag is encountered.

See also: [`AbstractAggregation`](@ref), [`AggregationStack`](@ref),
    [`SegmentedMax`](@ref), [`SegmentedSum`](@ref), [`SegmentedPNorm`](@ref), [`SegmentedLSE`](@ref).
"""
struct SegmentedMean{V <: AbstractVector{<:Number}} <: AbstractAggregation
    ψ::V
end

Flux.@layer :ignore SegmentedMean

SegmentedMean(T::Type, d::Int) = SegmentedMean(zeros(T, d))
SegmentedMean(d::Int) = SegmentedMean(Float32, d)

Flux.@forward SegmentedMean.ψ Base.getindex, Base.length, Base.size, Base.firstindex, Base.lastindex,
        Base.first, Base.last, Base.iterate, Base.eltype

Base.vcat(as::SegmentedMean...) = reduce(vcat, as |> collect)
function Base.reduce(::typeof(vcat), as::Vector{<:SegmentedMean})
    SegmentedMean(reduce(vcat, [a.ψ for a in as]))
end

function (a::SegmentedMean)(x::Maybe{AbstractMatrix{T}}, bags::AbstractBags,
                               w::Optional{AbstractVecOrMat{T}}=nothing) where T
    _check_agg(a, x)
    segmented_mean_forw(x, a.ψ, bags, w)
end

segmented_mean_forw(::Missing, ψ::AbstractVector, bags::AbstractBags, w) = repeat(ψ, 1, length(bags))
function segmented_mean_forw(x::AbstractMatrix, ψ::AbstractVector, bags::AbstractBags, w::Optional{AbstractVecOrMat}) 
    t = promote_type(eltype(x), eltype(ψ))
    y = zeros(t, size(x, 1), length(bags))
    @inbounds for (bi, b) in enumerate(bags)
        if isempty(b)
            for i in eachindex(ψ)
                y[i, bi] = ψ[i]
            end
        else
            for j in b
                for i in axes(x, 1)
                    y[i, bi] += _weight(w, i, j, t) * x[i, j]
                end
            end
            @views y[:, bi] ./= _bagnorm(w, b)
        end
    end
    y
end

function segmented_mean_back(Δ, y, x, ψ, bags, w) 
    dx = zero(x)
    dψ = zero(ψ)
    dw = isnothing(w) ? ZeroTangent() : zero(w)
    @inbounds for (bi, b) in enumerate(bags)
        if isempty(b)
            for i in eachindex(ψ)
                dψ[i] += Δ[i, bi]
            end
        else
            ws = _bagnorm(w, b)
            for j in b
                for i in axes(x, 1)
                    dx[i, j] += _weight(w, i, j, eltype(x)) * Δ[i, bi] / _weightsum(ws, i)
                    ∇dw_segmented_mean!(dw, Δ, x, y, w, ws, i, j, bi)
                end
            end
        end
    end
    dx, dψ, NoTangent(), dw
end

function segmented_mean_back(Δ, y, x::Missing, ψ, bags, w) 
    dψ = zero(ψ)
    @inbounds for bi in eachindex(bags)
        for i in eachindex(ψ)
            dψ[i] += Δ[i, bi]
        end
    end
    ZeroTangent(), dψ, NoTangent(), ZeroTangent()
end

∇dw_segmented_mean!(dw::ZeroTangent, Δ, x, y, w::Nothing, ws, i, j, bi) = nothing
function ∇dw_segmented_mean!(dw::AbstractVector, Δ, x, y, w::AbstractVector, ws, i, j, bi) 
    dw[j] += Δ[i, bi] * (x[i, j] - y[i, bi]) / ws
end
function ∇dw_segmented_mean!(dw::AbstractMatrix, Δ, x, y, w::AbstractMatrix, ws, i, j, bi)
    dw[i, j] += Δ[i, bi] * (x[i, j] - y[i, bi]) / ws[i]
end

function ChainRulesCore.rrule(::typeof(segmented_mean_forw), args...)
    y = segmented_mean_forw(args...)
    grad = Δ -> (NoTangent(), segmented_mean_back(Δ, y, args...)...)
    y, grad
end
