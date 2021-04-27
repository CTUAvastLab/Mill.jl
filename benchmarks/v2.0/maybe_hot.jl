using BenchmarkTools, Random
using Mill, Flux

using Plots, StatsPlots, LaTeXStrings; pgfplotsx()

experiments = [
    (
     label="DenseMatrix * (Full) OneHot",
     A_init = nc -> rand(Float32, 100, nc),
     B_init = (nc, nd) -> Flux.onehotbatch(rand(1:nc, nd), 1:nc)
    ), (
     label="DenseMatrix * Full MaybeHotMatrix",
     A_init = nc -> rand(Float32, 100, nc),
     B_init = (nc, nd) -> Mill.maybehotbatch(rand(1:nc, nd), 1:nc)
    ), (
     label="PostImputing * Full MaybeHotMatrix",
     A_init = nc -> rand(Float32, 100, nc) |> PostImputingMatrix,
     B_init = (nc, nd) -> Mill.maybehotbatch(rand(1:nc, nd), 1:nc)
    ), (
     label="PostImputing * Half Full MaybeHotMatrix",
     A_init = nc -> rand(Float32, 100, nc) |> PostImputingMatrix,
     B_init = (nc, nd) -> Mill.maybehotbatch([rand() < 0.5 ? missing : rand(1:nc) for _ in 1:nd], 1:nc)
    )
]

params = Base.Iterators.product([10, 50, 100], [100, 200, 300]) |> collect |> vec 
results = [Float64[] for _ in eachindex(experiments)]
for (nd, nc) in params
    @show nd, nc
    for (i,e) in experiments |> enumerate
        Random.seed!(0)
        A = e.A_init(nc)
        B = e.B_init(nc, nd)
        b = @benchmark $A * $B
        push!(results[i], median(b.times))
    end
end

xticklabels = [latexstring("$a|$b") for (a,b) in params]
groupedbar(xticklabels, hcat(results...); ylabel="ns", xlabel="Number of samples | Number of classes",
           labels=getindex.(experiments, :label) |> permutedims, title="A * B")
savefig("maybehot.png")

params = Base.Iterators.product([10, 50, 100], [100, 200, 300]) |> collect |> vec 
results = [Float64[] for _ in eachindex(experiments)]
for (nd, nc) in params
    @show nd, nc
    for (i,e) in experiments |> enumerate
        Random.seed!(0)
        A = e.A_init(nc)
        B = e.B_init(nc, nd)
        b = @benchmark gradient(A -> sum(A*$B), $A)
        push!(results[i], median(b.times))
    end
end

xticklabels = [latexstring("$a|$b") for (a,b) in params]
groupedbar(xticklabels, hcat(results...); ylabel="ns", xlabel="Number of samples | Number of classes",
           labels=getindex.(experiments, :label) |> permutedims, title="A * B (gradient)")
savefig("maybehot_grad.png")
