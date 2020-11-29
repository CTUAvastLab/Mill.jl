using BenchmarkTools, Random
using Mill, Flux

using Plots, LaTeXStrings; pgfplotsx()

Random.seed!(42)
params = Base.Iterators.product([50, 100, 300], [100, 200, 300]) |> collect |> vec 
for (m, bl) in params
    bags = length2bags(rand(0:bl, 20))
    A = rand(Float32, m, sum(length.(bags)))
    @show m, bl
    o = SegmentedMeanMax(m)
    @btime $o($A, $bags)
end

xticklabels = [latexstring("$a|$b") for (a,b) in params]

t1 = [37.764, 61.558, 144.835, 65.729, 88.520, 199.147, 91.258, 134.736, 433.920]
l1 = "Mill 1.5.2"
t2 = [38.907, 63.474, 146.922, 66.963, 89.890, 203.901, 89.999, 134.687, 424.865]
l2 = "Mill 2.0: bagcount on"
t3 = [36.856, 58.565, 138.007, 61.036, 80.387, 187.378, 89.691, 139.382, 427.770]
l3 = "Mill 2.0: bagcount off"
groupedbar(xticklabels, hcat(t1, t2, t3); ylabel=L"\mu s", xlabel="instance size | length of 20 bags", labels=[l1 l2 l3], title="MeanMax Aggregation")

savefig("aggregation.png")

Random.seed!(42)
params = Base.Iterators.product([50, 100, 300], [100, 200, 300]) |> collect |> vec 
for (m, bl) in params
    bags = length2bags(rand(0:bl, 20))
    A = rand(Float32, m, sum(length.(bags)))
    @show m, bl
    o = SegmentedMeanMax(m)
    @btime gradient((o, A) -> o(A, $bags) |> sum, $o, $A)
end

xticklabels = [latexstring("$a|$b") for (a,b) in params]

t1 = [217.039, 387.292, 1203, 357.977, 651.138, 1985, 518.316, 1120, 3796]
l1 = "Mill 1.5.2"
t2 = [217.585, 439.348, 1379, 371.837, 646.035, 1862, 459.842, 941.553, 3609]
l2 = "Mill 2.0: bagcount on"
t3 = [193.479, 398.941, 1144, 322.763, 590.045, 1627, 454.366, 1031, 3651]
l3 = "Mill 2.0: bagcount off"
groupedbar(xticklabels, hcat(t1, t2, t3); ylabel=L"\mu s", xlabel="instance size | length of 20 bags", labels=[l1 l2 l3], title="MeanMax Aggregation (gradients)")

savefig("aggregation_grad.png")
