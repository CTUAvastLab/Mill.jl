import Base: show

include("segmented_mean.jl")
include("segmented_max.jl")
include("segmented_pnorm.jl")
include("segmented_lse.jl")
export SegmentedMax, SegmentedMean, SegmentedMeanMax

const AGGF = [:segmented_max, :segmented_mean]
# generic code, for pnorm, situation is more complicated
for s in AGGF
    @eval $s(x::TrackedMatrix, args...) = Flux.Tracker.track($s, x, args...)
    @eval $s(x, bags, w::TrackedVector) = Flux.Tracker.track($s, x, bags, w)
    @eval $s(x::TrackedMatrix, bags, w::TrackedVector) = Flux.Tracker.track($s, x, bags, w)

    @eval $s(x::ArrayNode, args...) = mapdata(x -> $s(x, args...), x)

    @eval Flux.Tracker.@grad function $s(args...)
        $s(Flux.data.(args)...), Δ -> $(Symbol(string(s, "_back")))(Δ, args...)
    end
end

# with parameters
names = ["PNorm", "LSE", "Mean", "Max"]
fs = [:(PNorm(d)), :(LSE(d)), :segmented_mean, :segmented_max]
for idxs in powerset(collect(1:length(fs)))
    1 in idxs || 2 in idxs || continue
    @eval $(Symbol("Segmented", names[idxs]...))(d::Int) = Aggregation(tuple($(fs[idxs]...)))
    @eval export $(Symbol("Segmented", names[idxs]...))
end

const ParamAgg = Union{PNorm, LSE}

struct Aggregation{F}
    fs::F
end
Flux.@treelike Aggregation

Aggregation(a::Union{Function, ParamAgg}) = Aggregation((a,))
Aggregation(as::Union{Function, ParamAgg}...) = Aggregation(as)

(a::Aggregation)(args...) = vcat([f(args...) for f in a.fs]...)

function modelprint(io::IO, a::Aggregation; pad=[])
    paddedprint(io, "Aggregation($(join(a.fs, ", ")))\n")
end

function modelprint(io::IO, f; pad=[])
    paddedprint(io, "$f\n")
end
