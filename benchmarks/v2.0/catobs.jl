using Flux, Mill, BenchmarkTools, Random

X1 = Flux.onehotbatch([1,2,3,4,5], 1:10)
X2 = Mill.maybehotbatch([1,2,3,4,5], 1:10)
X3 = Mill.maybehotbatch(fill(missing, 5), 1:10)
X4 = Mill.maybehotbatch([1,missing,3,missing,5], 1:10)

copyinto(xs...; n=20÷length(xs)) = vcat([collect(deepcopy.(xs)) for _ in 1:n]...)

# should be equally fast
@btime hcat(X1, X1);
@btime hcat(X2, X2);
@btime hcat(X3, X3);
@btime hcat(X4, X4);

# slow, but possible
@btime hcat(X2, X4);
@btime hcat(X3, X4);

# should be equally fast
@btime reduce(hcat, $(copyinto(X1)));
@btime reduce(hcat, $(copyinto(X2)));
@btime reduce(hcat, $(copyinto(X3)));
@btime reduce(hcat, $(copyinto(X4)));

# slow, but possible
@btime reduce(hcat, $(copyinto(X2, X4)));
@btime reduce(hcat, $(copyinto(X3, X4)));

X1 = NGramMatrix([randstring(10) for _ in 1:10])
X2 = NGramMatrix(fill(missing, 10))
X3 = NGramMatrix([isodd(i) ? missing : randstring(10) for i in 1:10])

# should be equally fast
@btime hcat(X1, X1);
@btime hcat(X2, X2);
@btime hcat(X3, X3);

# slow, but possible
@btime hcat(X2, X3);

# should be equally fast
@btime reduce(hcat, $(copyinto(X1)));
@btime reduce(hcat, $(copyinto(X2)));
@btime reduce(hcat, $(copyinto(X3)));

# slow, but possible
@btime reduce(hcat, $(copyinto(X2, X3)));
