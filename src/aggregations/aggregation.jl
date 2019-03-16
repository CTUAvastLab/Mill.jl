import Base: show

include("segmented_mean.jl")
include("segmented_max.jl")
include("segmented_pnorm.jl")
include("segmented_lse.jl")
export SegmentedMax, SegmentedMean, SegmentedMeanMax

const AGGF = [:segmented_max, :segmented_mean]
# generic code, for pnorm, situation is more complicated
for s in AGGF
    # TODO what if the input is just Nothing in case of only empty bags?
    # TODO write tests
    # TODO metaprogram boilerplate code
    # TODO opravit inbounds vsude
    # TODO smazit masked verze?
    @eval $s(x::TrackedMatrix, args...) = Flux.Tracker.track($s, x, args...)
    @eval $s(x, bags, w::TrackedVector) = Flux.Tracker.track($s, x, bags, w)
    @eval $s(x::TrackedMatrix, bags, w::TrackedVector) = Flux.Tracker.track($s, x, bags, w)

    @eval $s(x::ArrayNode, args...) = mapdata(x -> $s(x, args...), x)

    @eval Flux.Tracker.@grad function $s(args...)
        $s(Flux.data.(args)...), Δ -> $(Symbol(string(s, "_back")))(Δ, args...)
    end
end

abstract type AggregationFunction end

struct Aggregation{N}
    fs::NTuple{N, AggregationFunction}
    Aggregation(fs::AggregationFunction...) = new(fs)
end

Flux.@treelike Aggregation

(a::Aggregation)(args...) = vcat([f(args...) for f in a.fs]...)

function modelprint(io::IO, a::Aggregation{N}; pad=[]) where N
    paddedprint(io, N == 1 ? "" : "⟨")
    for f in a.fs[1:end-1]
        modelprint(io, f, pad)
        paddedprint(io, ", ")
    end
    modelprint(io, a.fs[end], pad)
    paddedprint(io, (N == 1 ? "" : "⟩") * '\n')
end
