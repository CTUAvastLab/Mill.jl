import Base: show, getindex

abstract type AggregationFunction end

struct Aggregation{N} <: AggregationFunction
    fs::NTuple{N, AggregationFunction}
    Aggregation(fs::Vararg{AggregationFunction, N}) where N = new{N}(fs)
    Aggregation(fs::NTuple{N, AggregationFunction}) where N = new{N}(fs)
end

Flux.@treelike Aggregation

(a::Aggregation)(args...) = vcat([f(args...) for f in a.fs]...)

Base.show(io::IO, a::AggregationFunction) = modelprint(io, a)
Base.getindex(a::AggregationFunction, i) = a.fs[i]

function modelprint(io::IO, a::Aggregation{N}; pad=[]) where N
    paddedprint(io, N == 1 ? "" : "⟨")
    for f in a.fs[1:end-1]
        modelprint(io, f, pad=pad)
        paddedprint(io, ", ")
    end
    modelprint(io, a.fs[end], pad=pad)
    paddedprint(io, (N == 1 ? "" : "⟩"))
end

const MaybeMatrix = Union{AbstractMatrix, Missing}
const MaybeVector = Union{AbstractVector, Nothing}
const MaybeMask = Union{Vector{Bool}, Nothing}

bagnorm(w::Nothing, b) = length(b)
bagnorm(w::AbstractVector, b) = sum(view(w, b))
bagnorm(w::AbstractMatrix, b) = sum(view(w, :, b), dims=2)

# TODO delete
macro do_nothing()
    quote quote end end
end

# TODO delete
macro mask_rule(mask_type) 
    quote
        $(esc(mask_type)) <: Nothing ? $(@do_nothing) : :(!mask[bi] && continue)
    end
end

# TODO delete
macro fill_missing()
    quote quote return repeat(C, 1, length(bags)) end end
end

# TODO delete
complete_body(init_rule, empty_bag_update_rule, init_bag_rule, mask_rule,
              bag_update_rule, after_bag_rule, return_rule) = quote
    $init_rule
    for (j, b) in enumerate(bags)
        if isempty(b)
            for i in eachindex(C)
                @inbounds $empty_bag_update_rule
            end
        else
            @inbounds $init_bag_rule
            for bi in b
                @inbounds $mask_rule
                for i in 1:size(x, 1)
                    @inbounds $bag_update_rule
                end
            end
            @inbounds $after_bag_rule
        end
    end
    $return_rule
end

include("segmented_mean.jl")
include("segmented_max.jl")
include("segmented_pnorm.jl")
include("segmented_lse.jl")

export SegmentedMax, SegmentedMean, SegmentedPNorm, SegmentedLSE

const names = ["Mean", "Max", "PNorm", "LSE"]
for idxs in powerset(collect(1:length(names)))
    length(idxs) > 1 || continue
    for p in permutations(idxs)
        s = Symbol("Segmented", names[p]...)
        @eval function $s(d::Int)
            Aggregation($((Expr(:call, Symbol("Segmented" * n), :d)
                           for n in names[p])...))
        end
        @eval function $s(D::Vararg{Int, $(length(p))})
            Aggregation($((Expr(:call, Symbol("Segmented" * n), :(D[$i]))
                           for (i,n) in enumerate(names[p]))...))
        end
        @eval export $s
    end
end
