using Mill, BenchmarkTools, Serialization, ThreadsX, Zygote, Flux
using LinearAlgebra
(m, ds) = deserialize("/tmp/testdata.jls")
ps = Flux.params(m);
@btime gradient(() -> sum(m(ds).data), ps)
ccall((:openblas_get_num_threads64_, Base.libblas_name), Cint, ())
@btime gradient(() -> sum(m(ds).data), ps)

#Just 4 openblas threads, julia threadding disabled, 
# julia> @btime gradient(() -> sum(m(ds).data), ps)
#   1.189 s (16625 allocations: 765.25 MiB)
# Just 4 openblas threads, 8 threads in Julia
# julia> @btime gradient(() -> sum(m(ds).data), ps)
# 633.983 ms (20518 allocations: 765.46 MiB)
# Just 1 openblas threads, 8 threads in Julia
# julia> @btime gradient(() -> sum(m(ds).data), ps)
#  624.062 ms (20516 allocations: 765.46 MiB)


dss = [ds[i] for i in 1:Mill.nobs(ds)];
m(ds)
#time of the inference
@btime m(ds)
@btime ThreadsX.map(oneds -> m(oneds), dss)
#no threadding
# 588.425 ms (2387 allocations: 274.30 MiB)
#threadding inside Product Nodes
# 332.074 ms (4227 allocations: 274.40 MiB)
#threadding over samples
# 261.266 ms (98319 allocations: 279.36 MiB)

ps = Flux.params(m);
@btime gradient(() -> sum(m(ds).data), ps)


BUFFER_SIZE = 3_000_000_000
const buf = Array{UInt8}(undef, BUFFER_SIZE);
@btime NoGC.please!(buf) do 
	gradient(() -> sum(m(ds).data), ps)
end


gradient(x -> sum(map(i -> 2i, x)), [1,2,3])[1]
gradient(x -> sum(ThreadsX.map(i -> 2i, x)), [1,2,3])[1]

gst = gradient(() -> sum(m(ds).data), ps)

# with intra sample threadding 
# julia> @btime gradient(() -> sum(m(ds).data), ps)
# 677.039 ms (20539 allocations: 765.46 MiB)

# without threadding
# julia> @btime gradient(() -> sum(m(ds).data), ps)
# 1.490 s (16625 allocations: 765.25 MiB

# with threadding over samples
# julia> @btime ThreadsX.map(x -> gradient(() -> sum(m(x).data), ps), dss)
# 1.174 s (1355438 allocations: 2.24 GiB)

#Let's try to hack the multiplication
