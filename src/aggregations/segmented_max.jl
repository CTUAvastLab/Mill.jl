struct SegmentedMax{T} <: AggregationFunction
    C::T
end

SegmentedMax(d::Int) = SegmentedMax(zeros(Float32, d))
Flux.@treelike SegmentedMax

Base.show(io::IO, sm::SegmentedMax) = print(io, "SegmentedMax($(length(sm.C)))\n")
modelprint(io::IO, sm::SegmentedMax; pad=[]) = paddedprint(io, "SegmentedMax($(length(sm.C)))")

(m::SegmentedMax)(x::ArrayNode, args...) = mapdata(x -> m(x, args...), x)
(m::SegmentedMax)(x, args...) = segmented_max(x, m.C, args...)

segmented_max(::Missing, C::AbstractVector, bags, args...) = repeat(C, 1, length(bags)) 
function segmented_max(x::AbstractMatrix, C::AbstractVector, bags::AbstractBags, w = Fill(true, size(x,2)), mask = Fill(true, size(x,2))) 
    o = zeros(eltype(x), size(x, 1), length(bags))
    @inbounds for (j, b) in enumerate(bags)
        if isempty(b) || bagnormalization(w, b) == 0
                o[:, j] .= C
        else
            for bi in b
               for i in 1:size(x, 1)
                    o[i, j] = max(o[i, j], w[bi] * x[i, bi])
                end
            end
        end
    end
    o
end


function segmented_max_back(Δ, n, x, C, bags, w = nothing) 
    dx = zero(x)
    dC = zero(C)
    dw = (w == nothing) ? nothing : zero(w)
    v = similar(x, size(x, 1))
    idxs = zeros(Int, size(x, 1))
    for (j, b) in enumerate(bags)
        fill!(v, typemin(eltype(x)))
        if isempty(b) 
            dC .+= @view Δ[:, j]
        else
            @inbounds for bi in b
              for i in 1:size(x,1)
                if v[i] < x[i, bi]
                  idxs[i] = bi
                  v[i] = x[i, bi]
                end
              end
            end
          for i in 1:size(x, 1)
            dx[i, idxs[i]] = Δ[i, j]
          end
        end
    end
    (dx, dC, nothing, dw)
end

Zygote.@adjoint function segmented_max(args...)
    n = segmented_max(args...)
    grad = Δ -> segmented_max_back(Δ, n, args...)
    n, grad
end