include("segmented_mean.jl")
include("segmented_max.jl")
include("segmented_pnorm.jl")

const AGGF = [:segmented_max, :segmented_mean]
# generic code, for pnorm, situation is more complicated
for s in AGGF
    @eval $s(x::TrackedArray, args...) = Flux.Tracker.track($s, x, args...)
    @eval $s(x::ArrayNode, args...) = mapdata(x -> $s(x, args...), x)

    @eval Flux.Tracker.@grad function $s(x, args...)
        $s(Flux.data(x), Flux.data.(args)...), $(Symbol(string(s, "_back")))(x, args...)
    end
end

struct Aggregation
    fs
end

Aggregation(a::Union{Function, PNorm}) = Aggregation((a,))

Flux.@treelike Aggregation

(a::Aggregation)(args...) = vcat([f(args...) for f in a.fs]...)

segmented_meanmax = Aggregation((segmented_mean, segmented_max))
segmented_pnormmeanmax(d::Int) = Aggregation((PNorm(d), segmented_mean, segmented_max))
