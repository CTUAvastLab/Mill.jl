struct SegmentedMean{T, V <: AbstractVector{T}} <: AggregationOperator{T}
    ψ::V
end

Flux.@functor SegmentedMean

_SegmentedMean(d::Int) = SegmentedMean(zeros(Float32, d))

Flux.@forward SegmentedMean.ψ Base.getindex, Base.length, Base.size, Base.firstindex, Base.lastindex,
        Base.first, Base.last, Base.iterate, Base.eltype

Base.vcat(as::SegmentedMean...) = reduce(vcat, as |> collect)
function Base.reduce(::typeof(vcat), as::Vector{<:SegmentedMean})
    SegmentedMean(reduce(vcat, [a.ψ for a in as]))
end

function (m::SegmentedMean{T})(x::Maybe{AbstractMatrix{<:Maybe{T}}}, bags::AbstractBags,
                               w::Optional{AbstractVecOrMat{T}}=nothing) where T
    segmented_mean_forw(x, m.ψ, bags, w)
end
function (m::SegmentedMean{T})(x::AbstractMatrix{<:Maybe{T}}, bags::AbstractBags,
                               w::Optional{AbstractVecOrMat{T}}, mask::AbstractVector{T}) where T
    segmented_mean_forw(x .* mask', m.ψ, bags, w)
end

segmented_mean_forw(::Missing, ψ::AbstractVector, bags::AbstractBags, w) = repeat(ψ, 1, length(bags))
function segmented_mean_forw(x::AbstractMatrix, ψ::AbstractVector, bags::AbstractBags, w::Optional{AbstractVecOrMat}) 
    y = zeros(eltype(x), size(x, 1), length(bags))
    @inbounds for (bi, b) in enumerate(bags)
        if isempty(b)
            for i in eachindex(ψ)
                y[i, bi] = ψ[i]
            end
        else
            for j in b
                for i in 1:size(x, 1)
                    y[i, bi] += weight(w, i, j, eltype(x)) * x[i, j]
                end
            end
            @views y[:, bi] ./= bagnorm(w, b)
        end
    end
    y
end

function segmented_mean_back(Δ, y, x, ψ, bags, w) 
    dx = zero(x)
    dψ = zero(ψ)
    dw = isnothing(w) ? Zero() : zero(w)
    @inbounds for (bi, b) in enumerate(bags)
        if isempty(b)
            for i in eachindex(ψ)
                dψ[i] += Δ[i, bi]
            end
        else
            ws = bagnorm(w, b)
            for j in b
                for i in 1:size(x, 1)
                    dx[i, j] += weight(w, i, j, eltype(x)) * Δ[i, bi] / weightsum(ws, i)
                    ∇dw_segmented_mean!(dw, Δ, x, y, w, ws, i, j, bi)
                end
            end
        end
    end
    dx, dψ, DoesNotExist(), dw
end

function segmented_mean_back(Δ, y, x::Missing, ψ, bags, w) 
    dψ = zero(ψ)
    @inbounds for (bi, b) in enumerate(bags)
        for i in eachindex(ψ)
            dψ[i] += Δ[i, bi]
        end
    end
    Zero(), dψ, DoesNotExist(), Zero()
end

∇dw_segmented_mean!(dw::Zero, Δ, x, y, w::Nothing, ws, i, j, bi) = nothing
function ∇dw_segmented_mean!(dw::AbstractVector, Δ, x, y, w::AbstractVector, ws, i, j, bi) 
    dw[j] += Δ[i, bi] * (x[i, j] - y[i, bi]) / ws
end
function ∇dw_segmented_mean!(dw::AbstractMatrix, Δ, x, y, w::AbstractMatrix, ws, i, j, bi)
    dw[i, j] += Δ[i, bi] * (x[i, j] - y[i, bi]) / ws[i]
end

function rrule(::typeof(segmented_mean_forw), args...)
    y = segmented_mean_forw(args...)
    grad = Δ -> (NO_FIELDS, segmented_mean_back(Δ, y, args...)...)
    y, grad
end
