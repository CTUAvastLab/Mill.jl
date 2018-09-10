include("segmented_mean.jl")
include("segmented_max.jl")

aggregation_vcat(fs...) = (args...) -> vcat([f(args...) for f in fs]...)
segmented_meanmax = aggregation_vcat(segmented_mean, segmented_max)
