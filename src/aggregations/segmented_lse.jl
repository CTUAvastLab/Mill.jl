"""
    SegmentedLSE{T, V <: AbstractVector{T}} <: AbstractAggregation{T}

[`AbstractAggregation`](@ref) implementing segmented log-sum-exp (LSE) aggregation:

``
f(\\{x_1, \\ldots, x_k\\}; r) = \\frac{1}{r}\\log \\left(\\frac{1}{k} \\sum_{i = 1}^{k} \\exp({r\\cdot x_i})\\right)
``

Stores a vector of parameters `ψ` that are filled into the resulting matrix in case an empty bag is encountered,
and a vector of parameters `r` used during computation.

See also: [`AbstractAggregation`](@ref), [`AggregationStack`](@ref), [`lse_aggregation`](@ref),
    [`SegmentedMax`](@ref), [`SegmentedMean`](@ref), [`SegmentedSum`](@ref), [`SegmentedPNorm`](@ref).
"""
struct SegmentedLSE{T <: AbstractFloat, V <: AbstractVector{T}} <: AbstractAggregation{T}
    ψ::V
    ρ::V
end

Flux.@functor SegmentedLSE

SegmentedLSE{T}(d::Int) where T = SegmentedLSE(zeros(T, d), randn(T, d))
SegmentedLSE(T::Type, d::Int) = SegmentedLSE{T}(d)
SegmentedLSE(d::Int) = SegmentedLSE(Float32, d)

Flux.@forward SegmentedLSE.ψ Base.getindex, Base.length, Base.size, Base.firstindex, Base.lastindex,
        Base.first, Base.last, Base.iterate, Base.eltype

Base.vcat(as::SegmentedLSE...) = reduce(vcat, as |> collect)
function Base.reduce(::typeof(vcat), as::Vector{<:SegmentedLSE})
    SegmentedLSE(reduce(vcat, [a.ψ for a in as]),
                 reduce(vcat, [a.ρ for a in as]))
end

r_map(ρ) = @. softplus(ρ)
inv_r_map(r) = @. relu(r) + log1p(-exp(-abs(r)))

function (m::SegmentedLSE)(x::Maybe{AbstractMatrix{T}}, bags::AbstractBags,
                              w::Optional{AbstractVecOrMat{T}}=nothing) where T
    segmented_lse_forw(x, m.ψ, r_map(m.ρ), bags)
end

function _lse_precomp(x::AbstractMatrix, r, bags)
    T = promote_type(eltype(x), eltype(r))
    M = zeros(T, length(r), length(bags))
    @inbounds for (bi, b) in enumerate(bags)
        if !isempty(b)
            for i in eachindex(r)
                M[i, bi] = r[i] * x[i, first(b)]
            end
            for j in b[2:end]
                for i in eachindex(r)
                    M[i, bi] = max(M[i, bi], r[i] * x[i, j])
                end
            end
        end
    end
    M
end

function _segmented_lse_norm(x::AbstractMatrix, ψ, r, bags::AbstractBags, M)
    t = promote_type(eltype(x), eltype(ψ))
    y = zeros(t, length(r), length(bags))
    @inbounds for (bi, b) in enumerate(bags)
        if isempty(b)
            for i in eachindex(ψ)
                y[i, bi] = ψ[i]
            end
        else
            for j in b
                for i in eachindex(r)
                    y[i, bi] += exp.(r[i] * x[i, j] - M[i, bi])
                end
            end
            for i in eachindex(r)
                y[i, bi] = (log(y[i, bi]) - log(length(b)) + M[i, bi]) / r[i]
            end
        end
    end
    y
end

segmented_lse_forw(::Missing, ψ::AbstractVector, r, bags::AbstractBags) = repeat(ψ, 1, length(bags))
function segmented_lse_forw(x::AbstractMatrix, ψ, r, bags::AbstractBags)
    M = _lse_precomp(x, r, bags)
    _segmented_lse_norm(x, ψ, r, bags, M)
end

function segmented_lse_back(Δ, y, x, ψ, r, bags, M)
    dx = zero(x)
    dψ = zero(ψ)
    dr = zero(r)
    s1 = zeros(eltype(x), length(r))
    s2 = zeros(eltype(x), length(r))

    @inbounds for (bi, b) in enumerate(bags)
        if isempty(b)
            for i in eachindex(ψ)
                dψ[i] += Δ[i, bi]
            end
        else
            for i in eachindex(r)
                s1[i] = s2[i] = zero(eltype(x))
            end
            for j in b
                for i in eachindex(r)
                    e = exp(r[i] * x[i, j] - M[i, bi])
                    s1[i] += e
                    s2[i] += x[i, j] * e
                end
            end
            for j in b
                for i in eachindex(r)
                    dx[i, j] = Δ[i, bi] * exp(r[i] * x[i, j] - M[i, bi]) / s1[i]
                end
            end
            for i in eachindex(r)
                dr[i] += Δ[i, bi] * (s2[i]/s1[i] - y[i, bi]) / r[i]
            end
        end
    end
    dx, dψ, dr, DoesNotExist(), Zero()
end

function segmented_lse_back(Δ, ::Missing, ψ, bags)
    dψ = zero(ψ)
    @inbounds for (bi, b) in enumerate(bags)
        for i in eachindex(ψ)
            dψ[i] += Δ[i, bi]
        end
    end
    Zero(), dψ, Zero(), DoesNotExist(), Zero()
end

function ChainRulesCore.rrule(::typeof(segmented_lse_forw),
        x::AbstractMatrix, ψ::AbstractVector, r::AbstractVector, bags::AbstractBags)
    M = _lse_precomp(x, r, bags)
    y = _segmented_lse_norm(x, ψ, r, bags, M)
    grad = Δ -> (NO_FIELDS, segmented_lse_back(Δ, y, x, ψ, r, bags, M)...)
    y, grad
end

function segmented_lse_forw(::typeof(segmented_lse_forw), x::Missing, ψ::AbstractVector, r::AbstractVector, bags::AbstractBags)
    y = segmented_lse_forw(x, ψ, r, bags)
    grad = Δ -> (NO_FIELDS, segmented_lse_back(Δ, x, ψ, bags)...)
    y, grad
end
