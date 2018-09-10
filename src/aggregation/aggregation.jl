using Combinatorics: permutations
using IterTools: subsets

include("segmented_mean.jl")
include("segmented_max.jl")

aggregation_vcat(fs...) = (args...) -> vcat([f(args...) for f in fs]...)

const AGG = [:max, :mean, :pnorm]

# generate all possible combinations
for s in subsets(AGG)
    length(s) > 1 && for p in permutations(s)
        @eval $(Symbol(string("segmented_", p...))) = aggregation_vcat($(map(a -> Symbol(string("segmented_", a)), p)...))
    end
end
