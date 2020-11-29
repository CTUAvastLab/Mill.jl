using BenchmarkTools, Random
using Mill, Flux

using Plots, LaTeXStrings; pgfplotsx()

Random.seed!(42)
params = Base.Iterators.product([10, 50, 100], [100, 200, 300]) |> collect |> vec 
for (nd, nc) in params
    @show nd, nc
    # A = rand(Float32, 100, nc)
    # B = Flux.onehotbatch(rand(1:nc, nd), 1:nc)
    # A = PostImputingMatrix(rand(Float32, 100, nc))
    # B = Mill.maybehotbatch(rand(1:nc, nd), 1:nc)
    # B = Mill.maybehotbatch([rand() < 0.5 ? missing : rand(1:nc) for _ in 1:nd], 1:nc)
    @btime $A * $B
end

xticklabels = [latexstring("$a|$b") for (a,b) in params]

t1 = [1.626, 5.422, 10.291, 1.587, 5.589, 10.744, 1.635, 5.513, 11.026]
l1 = "DenseMatrix * OneHotMatrix"
t2 = [884.980/1000, 3.915, 7.769, 847.077/1000, 3.941, 7.631, 875.275/1000, 3.962, 7.935]
l2 = "PostImputing * Full MaybeHotMatrix"
t3 = [587.663/1000, 2.396, 4.646, 595.070/1000, 2.711, 4.623, 669.864/1000, 2.433, 4.603]
l3 = "PostImputing * Half Full MaybeHotMatrix"
groupedbar(xticklabels, hcat(t1, t2, t3); ylabel=L"\mu s", xlabel="Number of samples | Number of classes", labels=[l1 l2 l3], title="PostImputing MaybeHot")

savefig("maybehot.png")

Random.seed!(42)
params = Base.Iterators.product([10, 50, 100], [100, 200, 300]) |> collect |> vec 
for (nd, nc) in params
    @show nd, nc
    A = rand(Float32, 100, nc)
    B = Flux.onehotbatch(rand(1:nc, nd), 1:nc)
    # A = PostImputingMatrix(rand(Float32, 100, nc))
    # B = Mill.maybehotbatch(rand(1:nc, nd), 1:nc)
    # B = Mill.maybehotbatch([rand() < 0.5 ? missing : rand(1:nc) for _ in 1:nd], 1:nc)
    @btime gradient((A,B) -> sum(A*B), $A, $B)
end

t1 = [22.656, 26.218, 40.337, 37.472, 35.936, 51.702, 31.317, 43.944, 65.023]
l1 = "DenseMatrix * OneHotMatrix"
t2 = [5.543, 12.771, 27.913, 8.151, 18.499, 24.678, 10.754, 20.389, 32.954]
l2 = "PostImputing * Full MaybeHotMatrix"
t3 = [4.996, 7.876, 11.332, 8.095, 11.329, 14.038, 9.755, 13.343, 16.022]
l3 = "PostImputing * Half Full MaybeHotMatrix"
groupedbar(xticklabels, hcat(t1, t2, t3); ylabel=L"\mu s", xlabel="Number of samples | Number of classes", labels=[l1 l2 l3], title="PostImputing MaybeHot (gradient)")

savefig("maybehot_grad.png")
