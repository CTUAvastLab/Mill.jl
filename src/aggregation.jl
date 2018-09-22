include("segmented_mean.jl")
include("segmented_max.jl")
include("segmented_pnorm.jl")
include("segmented_lse.jl")

const AGGF = [:segmented_max, :segmented_mean]
# generic code, for pnorm, situation is more complicated
for s in AGGF
    @eval $s(x::TrackedArray, args...) = Flux.Tracker.track($s, x, args...)
    @eval $s(x::ArrayNode, args...) = mapdata(x -> $s(x, args...), x)

    @eval Flux.Tracker.@grad function $s(x, args...)
        $s(Flux.data(x), Flux.data.(args)...), $(Symbol(string(s, "_back")))(x, args...)
    end
end

const ParamAgg = Union{PNorm, LSE}

struct Aggregation
    fs
end
Flux.@treelike Aggregation

Aggregation(a::Union{Function, ParamAgg}) = Aggregation((a,))
(a::Aggregation)(args...) = vcat([f(args...) for f in a.fs]...)

# convenience definitions - nested Aggregations work, but call definitions directly to avoid overhead
# without parameters
SegmentedMax() = Aggregation(segmented_max)
SegmentedMean() = Aggregation(segmented_mean)
SegmentedMeanMax() = Aggregation((segmented_mean, segmented_max))
for s in [:SegmentedMax, :SegmentedMean, :SegmentedMeanMax]
    @eval $s(d::Int) = $s()
end

# with parameters
names = ["PNorm", "LSE", "Mean", "Max"]
fs = [:(PNorm(d)), :(LSE(d)), :segmented_mean, :segmented_max]
for idxs in powerset(collect(1:length(fs)))
    1 in idxs || 2 in idxs || continue
    @eval $(Symbol("Segmented", names[idxs]...))(d::Int) = Aggregation(tuple($(fs[idxs]...)))
end
