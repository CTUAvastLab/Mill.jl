"""
    SegmentedSum{V <: AbstractVector{<:Number}} <: AbstractAggregation

[`AbstractAggregation`](@ref) implementing segmented sum aggregation:

``
f(\\{x_1, \\ldots, x_k\\}) = \\sum_{i = 1}^{k} x_i
``

Stores a vector of parameters `ψ` that are filled into the resulting matrix in case an empty bag is encountered.

See also: [`AbstractAggregation`](@ref), [`AggregationStack`](@ref), [`sum_aggregation`](@ref),
    [`SegmentedMax`](@ref), [`SegmentedMean`](@ref), [`SegmentedPNorm`](@ref), [`SegmentedLSE`](@ref).
"""
struct SegmentedSum{V <: AbstractVector{<:Number}} <: AbstractAggregation
    ψ::V
end

Flux.@functor SegmentedSum

SegmentedSum(T::Type, d::Int) = SegmentedSum(zeros(T, d))
SegmentedSum(d::Int) = SegmentedSum(Float32, d)

Flux.@forward SegmentedSum.ψ Base.getindex, Base.length, Base.size, Base.firstindex, Base.lastindex,
        Base.first, Base.last, Base.iterate, Base.eltype

Base.vcat(as::SegmentedSum...) = reduce(vcat, as |> collect)
function Base.reduce(::typeof(vcat), as::Vector{<:SegmentedSum})
    SegmentedSum(reduce(vcat, [a.ψ for a in as]))
end

function (a::SegmentedSum)(x::Maybe{AbstractMatrix{T}}, bags::AbstractBags,
                              w::Optional{AbstractVecOrMat{T}}=nothing) where T
    _check_agg(a, x)
    segmented_sum_forw(x, a.ψ, bags, w)
end

segmented_sum_forw(::Missing, ψ::AbstractVector, bags::AbstractBags, w) = repeat(ψ, 1, length(bags))
function segmented_sum_forw(x::AbstractMatrix, ψ::AbstractVector, bags::AbstractBags, w::Nothing) 
    t = promote_type(eltype(x), eltype(ψ))
    y = zeros(t, size(x, 1), length(bags))
    @inbounds for (bi, b) in enumerate(bags)
        if isempty(b)
            @turbo for i in eachindex(ψ)
                y[i, bi] = ψ[i]
            end
        else
            for j in b
                @turbo for i in 1:size(x, 1)
                    y[i, bi] += x[i, j]
                end
            end
        end
    end
    y
end
function segmented_sum_forw(x::AbstractMatrix, ψ::AbstractVector, bags::AbstractBags, w::AbstractVector) 
    t = promote_type(eltype(x), eltype(ψ))
    y = zeros(t, size(x, 1), length(bags))
    @inbounds for (bi, b) in enumerate(bags)
        if isempty(b)
            @turbo for i in eachindex(ψ)
                y[i, bi] = ψ[i]
            end
        else
            for j in b
                @turbo for i in 1:size(x, 1)
                    y[i, bi] += w[j] * x[i, j]
                end
            end
        end
    end
    y
end
function segmented_sum_forw(x::AbstractMatrix, ψ::AbstractVector, bags::AbstractBags, w::AbstractMatrix)
    t = promote_type(eltype(x), eltype(ψ))
    y = zeros(t, size(x, 1), length(bags))
    @inbounds for (bi, b) in enumerate(bags)
        if isempty(b)
            @turbo for i in eachindex(ψ)
                y[i, bi] = ψ[i]
            end
        else
            for j in b
                @turbo for i in 1:size(x, 1)
                    y[i, bi] += w[i, j] * x[i, j]
                end
            end
        end
    end
    y
end

function segmented_sum_back(Δ, y, x, ψ, bags, w) 
    dx = similar(x)
    dψ = zero(ψ)
    dw = isnothing(w) ? ZeroTangent() : zero(w)
    @inbounds for (bi, b) in enumerate(bags)
        if isempty(b)
            @turbo for i in eachindex(ψ)
                dψ[i] += Δ[i, bi]
            end
        else
            ∇dxdw_segmented_sum!(dx, dw, Δ, x, y, w, b, bi)
            #for j in b
            #    for i in 1:size(x, 1)
            #        dx[i, j] = _weight(w, i, j, eltype(x)) * Δ[i, bi]
            #        ∇dw_segmented_sum!(dw, Δ, x, y, w, i, j, bi)
            #    end
            #end
        end
    end
    dx, dψ, NoTangent(), dw
end

function segmented_sum_back(Δ, y, x::Missing, ψ, bags, w::Nothing) 
    dψ = zero(ψ)
    @inbounds for (bi, b) in enumerate(bags)
        @turbo for i in eachindex(ψ)
            dψ[i] += Δ[i, bi]
        end
    end
    ZeroTangent(), dψ, NoTangent(), ZeroTangent()
end

function ∇dxdw_segmented_sum!(dx, dw::ZeroTangent, Δ, x, y, w::Nothing, b, bi)
    for j in b
        @turbo for i in 1:size(x, 1)
            dx[i, j] = Δ[i, bi]
        end
    end
end
function ∇dxdw_segmented_sum!(dx, dw::AbstractVector, Δ, x, y, w::AbstractVector, b, bi)
    for j in b
        for i in 1:size(x, 1)
            dw[j] += Δ[i, bi] * (x[i, j])
        end
        @turbo for i in 1:size(x, 1)
            dx[i, j] = w[j] * Δ[i, bi]
        end
    end
end
function ∇dxdw_segmented_sum!(dx, dw::AbstractMatrix, Δ, x, y, w::AbstractMatrix, b, bi)
    for j in b
        @turbo for i in 1:size(x, 1)
            dw[i, j] += Δ[i, bi] * (x[i, j])
            dx[i, j] = w[i, j] * Δ[i, bi]
        end
    end
end
#∇dw_segmented_sum!(dw::ZeroTangent, Δ, x, y, w::Nothing, i, j, bi) = nothing
#function ∇dw_segmented_sum!(dw::AbstractVector, Δ, x, y, w::AbstractVector, i, j, bi) 
#    dw[j] += Δ[i, bi] * (x[i, j])
#end
#function ∇dw_segmented_sum!(dw::AbstractMatrix, Δ, x, y, w::AbstractMatrix, i, j, bi)
#    dw[i, j] += Δ[i, bi] * (x[i, j])
#end

function ChainRulesCore.rrule(::typeof(segmented_sum_forw), args...)
    y = segmented_sum_forw(args...)
    grad = Δ -> (NoTangent(), segmented_sum_back(Δ, y, args...)...)
    y, grad
end
