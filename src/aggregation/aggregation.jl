using Combinatorics: permutations
using IterTools: subsets

include("segmented_mean.jl")
include("segmented_max.jl")
include("segmented_pnorm.jl")

aggregation_vcat(fs...) = (args...) -> vcat([f(args...) for f in fs]...)

const AGG = [:max, :mean, :pnorm]

# generate all possible combinations
for s in subsets(AGG)
    length(s) > 1 && for p in permutations(s)
        @eval $(Symbol(string("segmented_", p...))) = aggregation_vcat($(map(a -> Symbol(string("segmented_", a)), p)...))
    end
end

const AGGF = [:segmented_max, :segmented_mean, :segmented_pnorm]
# generic code
for s in AGGF
    @eval $s(x::Flux.Tracker.TrackedArray, args...) = Flux.Tracker.track($s, x, args...)
    @eval $s(x::ArrayNode, args...) = ArrayNode($s(x.data, args...))

    s != :segmented_pnorm && @eval Flux.Tracker.@grad function $s(x, args...)
    	 $s(Flux.data(x), Flux.data.(args)...), $(Symbol(string(s, "_back")))(x, args...)
    end
end
