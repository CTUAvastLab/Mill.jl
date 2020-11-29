using BenchmarkTools, Random
using Mill, Flux

using StatsPlots, LaTeXStrings; pgfplotsx()

Random.seed!(42)
params = Base.Iterators.product([10, 100, 1000], [100, 200, 300]) |> collect |> vec 
for (sl,sn) in params
    @show sn, sl
    # A = rand(100, 2053)
    A = PostImputingMatrix(rand(100, 2053))
    B = NGramMatrix([rand() < 0.5 ? missing : randstring(sl) for _ in 1:sn], 3, 256, 2053)
    # B = NGramMatrix(fill(missing, sn), 3, 256, 2053)
    # B = NGramMatrix([randstring(sl) for _ in 1:sn], 3, 256, 2053)
    @btime *($A, $B);
end

xticklabels = [latexstring("$a|$b") for (a,b) in params]

t1 = [56.427, 425.149, 4.043*1000, 112.646, 850.431, 8.274*1000, 169.531, 1.274*1000, 12.801*1000]
l1 = "Mill 1.5.2: Matrix * NGramMatrix{String, …}"
t2 = [55.109, 440.126, 4*1000, 116.665, 860.89, 8.186*1000, 183.248, 1.294*1000, 13.093*1000]
l2 = "Mill 2.0: Matrix * NGramMatrix{String, …}"
t4 = [38.523, 38.552, 38.452, 38.420, 38.650, 38.565, 37.717, 38.652, 38.789]
l4 = "Mill 2.0: PostImputingMatrix * NGramMatrix{Missing, …}"
t3 = [37.686, 256.430, 2179, 87.778, 450.185, 4497, 110.967, 773.528, 6761]
l3 = "Mill 2.0: PostImputingMatrix * NGramMatrix{Union{Missing, String}, …}"
groupedbar(xticklabels, hcat(t1, t2, t3, t4); ylabel=L"\mu s", xlabel="n. of strings | string length", labels=[l1 l2 l3 l4], title="PostImputing NGramMatrix multiplication")

savefig("ngram_mult.png")

Random.seed!(42)
params = Base.Iterators.product([10, 100, 1000], [100, 200, 300]) |> collect |> vec 
for (sl,sn) in params
    @show sn, sl
    # A = rand(100, 2053)
    A = PostImputingMatrix(rand(100, 2053))
    # B = NGramMatrix([rand() < 0.5 ? missing : randstring(sl) for _ in 1:sn], 3, 256, 2053)
    B = NGramMatrix(fill(missing, sn), 3, 256, 2053)
    # B = NGramMatrix([randstring(sl) for _ in 1:sn], 3, 256, 2053)
    @btime gradient((A,B) -> sum(A*B), $A, $B)
end

xticklabels = [latexstring("$a|$b") for (a,b) in params]

t1 = [280.452, 1104, 8696, 442.584, 1848, 16047, 541.064, 2754, 25244]
l1 = "Mill 1.5.2: Matrix * NGramMatrix{String, …}"
t2 = [287.431, 1108, 9185, 421.466, 2061, 15871, 560.260, 2944, 28016]
l2 = "Mill 2.0: Matrix * NGramMatrix{String, …}"
t4 = [28.615, 28.226, 29.704, 52.057, 49.217, 54.538, 82.527, 77.738, 72.033]
l4 = "Mill 2.0: PostImputingMatrix * NGramMatrix{Missing, …}"
t3 = [201.775, 843.369, 4098, 370.220, 1302, 10952, 506.579, 1668, 16352]
l3 = "Mill 2.0: PostImputingMatrix * NGramMatrix{Union{Missing, String}, …}"
groupedbar(xticklabels, hcat(t1, t2, t3, t4); ylabel=L"\mu s", xlabel="n. of strings | string length", labels=[l1 l2 l3 l4], title="PostImputing NGramMatrix multiplication (grad)")

savefig("ngram_mult_grad.png")
