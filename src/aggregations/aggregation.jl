import Base: show

# TODO Float32 x Float64
#     # TODO what if the input is just Nothing in case of only empty bags? - reseno v BagNode
#     # TODO write tests na C
#     # TODO write tests for nothing input
#
abstract type AggregationFunction end

struct Aggregation{N} <: AggregationFunction
    fs::NTuple{N, AggregationFunction}
    Aggregation(fs::Vararg{AggregationFunction, N}) where N = new{N}(fs)
    Aggregation(fs::NTuple{N, AggregationFunction}) where N = new{N}(fs)
end

Flux.@treelike Aggregation

(a::Aggregation)(args...) = vcat([f(args...) for f in a.fs]...)

Base.show(io::IO, a::AggregationFunction) = modelprint(io, a)

function modelprint(io::IO, a::Aggregation{N}; pad=[]) where N
    paddedprint(io, N == 1 ? "" : "⟨")
    for f in a.fs[1:end-1]
        modelprint(io, f, pad=pad)
        paddedprint(io, ", ")
    end
    modelprint(io, a.fs[end], pad=pad)
    paddedprint(io, (N == 1 ? "" : "⟩") * '\n')
end

const Tracked = Union{TrackedMatrix, TrackedVector}
const MaybeMatrix = Union{AbstractMatrix, Nothing}
const MaybeVector = Union{AbstractVector, Nothing}
const MaybeMask = Union{Vector{Bool}, Nothing}

macro do_nothing()
    quote quote end end
end

macro mask_rule(mask_type) 
    quote
        $(esc(mask_type)) <: Nothing ? $(@do_nothing) : :(!mask[bi] && continue)
    end
end

complete_body(init_rule, empty_bag_update_rule, init_bag_rule, mask_rule,
              bag_update_rule, after_bag_rule, return_rule) = quote
    $init_rule
    for (j, b) in enumerate(bags)
        if isempty(b)
            for i in 1:size(x, 1)
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
        # generates calls like
        # SegmentedMeanMax(d::Int) = Aggregation(SegmentedMean(d), SegmentedMax(d))
        # SegmentedMeanMax(d::Int) = Aggregation(SegmentedMean(d), SegmentedMax(d))
        @eval function $s(d::Int)
            Aggregation($(
                          (
                           map(names[p]) do n
                               s_call = Symbol("Segmented" * n)
                               :($s_call(d))
                           end
                          )
                          ...))
        end
        # generates calls like
        # SegmentedMeanMax(d1::Int, d2::Int) = Aggregation(SegmentedMean(d1), SegmentedMax(d2))
        @eval function $s(D::Vararg{Int, $(length(p))})
            Aggregation($(
                          (
                           map(enumerate(names[p])) do (i, n)
                               s_call = Symbol("Segmented" * n)
                               :($s_call(D[$i]))
                           end
                          )
                          ...))
        end
        @eval export $s
    end
end

