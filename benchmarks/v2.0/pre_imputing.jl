using BenchmarkTools, Random
using Mill, Flux

using Plots, LaTeXStrings; pgfplotsx()

Random.seed!(42)
params = Base.Iterators.product([100, 200, 300], [100, 200, 300]) |> collect |> vec 
for (m, n) in params
    @show m, n
    # A = rand(Float32, m, n)
    A = PreImputingMatrix(rand(Float32, m, n))
    # B = rand(Float32, n, m)
    B = [rand() < 0.5 ? missing : rand() for _ in 1:n, _ in 1:m]
    @btime $A * $B
end

xticklabels = [latexstring("$a|$b") for (a,b) in params]

t1 = [22.859, 50.551, 95.680, 38.428, 82.887, 200.255, 54.975, 133.416, 232.263] ./ 1000
l1 = "matrix(m,n) * Full Matrix(n,m)"
t2 = [3.868, 17.397, 39.650, 8.290, 35.292, 78.747, 13.223, 53.783, 121.800]
l2 = "Matrix(m,n) * Half Full Matrix(n,m)"
t3 = [22.114, 59.371, 96.551, 29.781, 67.402, 208.026, 36.066, 127.561, 227.967] ./ 1000
l3 = "Preimputingmatrix(m,n) * Full Matrix(n,m)"
t4 = [55.889, 119.639, 196.984, 94.915, 204.092, 335.075, 136.566, 298.371, 487.540] ./ 1000
l4 = "PreImputingmatrix(m,n) * Half Full Matrix(n,m)"
groupedbar(xticklabels, hcat(t1, t2, t3, t4); ylabel=L"ms", xlabel="n | m", labels=[l1 l2 l3 l4], title="PreImputing (log y)", yscale=:log10)

savefig("pre_imputing.png")

Random.seed!(42)
params = Base.Iterators.product([100, 200, 300], [100, 200, 300]) |> collect |> vec 
for (m, n) in params
    @show m, n
    # A = rand(Float32, m, n)
    A = PreImputingMatrix(rand(Float32, m, n))
    # B = rand(Float32, n, m)
    B = [rand() < 0.5 ? missing : rand() for _ in 1:n, _ in 1:m]
    @btime gradient((A, B) -> sum(A*B), $A, $B)
end

xticklabels = [latexstring("$a|$b") for (a,b) in params]

t1 = [901.895 / 1000, 3.553, 7.889, 1.883, 7.232, 16.162, 2.648, 10.688, 24.344]
l1 = "Matrix(m,n) * Full Matrix(n,m)"
t2 = [909.361 / 1000, 3.650, 7.865, 1.792, 7.260, 16.200, 2.782, 10.598, 24.071]
l2 = "PreImputingMatrix(m,n) * Full Matrix(n,m)"
t3 = [1.077, 4.042, 9.111, 2.135, 7.637, 17.257, 3.075, 12.057, 24.922]
l3 = "PreImputingMatrix(m,n) * Half Full Matrix(n,m)"
groupedbar(xticklabels, hcat(t1, t2, t3); ylabel=L"ms", xlabel="n | m", labels=[l1 l2 l3], title="PreImputing (grad)")

savefig("pre_imputing_grad.png")

