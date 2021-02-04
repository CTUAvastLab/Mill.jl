struct SegmentedMax{T, V <: AbstractVector{T}} <: AggregationOperator{T}
    ψ::V
end

Flux.@functor SegmentedMax

_SegmentedMax(d::Int) = SegmentedMax(zeros(Float32, d))

Flux.@forward SegmentedMax.ψ Base.getindex, Base.length, Base.size, Base.firstindex, Base.lastindex,
        Base.first, Base.last, Base.iterate, Base.eltype

Base.vcat(as::SegmentedMax...) = reduce(vcat, as |> collect)
function Base.reduce(::typeof(vcat), as::Vector{<:SegmentedMax})
    SegmentedMax(reduce(vcat, [a.ψ for a in as]))
end

function (m::SegmentedMax{T})(x::Maybe{AbstractMatrix{<:Maybe{T}}}, bags::AbstractBags,
                              w::Optional{AbstractVecOrMat{T}}=nothing) where T
    segmented_max_forw(x, m.ψ, bags)
end
function (m::SegmentedMax{T})(x::AbstractMatrix{<:Maybe{T}}, bags::AbstractBags,
                              w::Optional{AbstractVecOrMat{T}}, mask::AbstractVector{T}) where T
    segmented_max_forw(x .+ _typemin(T) * mask', m.ψ, bags)
end

segmented_max_forw(::Missing, ψ::AbstractVector, bags::AbstractBags) = repeat(ψ, 1, length(bags))
function segmented_max_forw(x::AbstractMatrix, ψ::AbstractVector, bags::AbstractBags) 
    y = fill(_typemin(eltype(x)), size(x, 1), length(bags)) |> typeof(x)
    @inbounds for (bi, b) in enumerate(bags)
        if isempty(b)
            for i in eachindex(ψ)
                y[i, bi] = ψ[i]
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

function segmented_max_back(Δ, y, x, ψ, bags) 
    dx = zero(x)
    dψ = zero(ψ)
    v = similar(x, size(x, 1))
    idxs = zeros(Int, size(x, 1))
    @inbounds for (bi, b) in enumerate(bags)
        if isempty(b)
            for i in eachindex(ψ)
                dψ[i] += Δ[i, bi]
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
                dx[i, idxs[i]] += Δ[i, bi]
            end
        end
    end
    dx, dψ, DoesNotExist()
end

function segmented_max_back(Δ, y, x::Missing, ψ, bags) 
    dψ = zero(ψ)
    @inbounds for (bi, b) in enumerate(bags)
        for i in eachindex(ψ)
            dψ[i] += Δ[i, bi]
        end
    end
    Zero(), dψ, DoesNotExist()
end

function ChainRulesCore.rrule(::typeof(segmented_max_forw), args...)
    y = segmented_max_forw(args...)
    grad = Δ -> (NO_FIELDS, segmented_max_back(Δ, y, args...)...)
    y, grad
end
