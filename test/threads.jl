using Mill, BenchmarkTools, Serialization, Zygote, Flux
using LinearAlgebra
using Setfield
using ThreadPools
using Profile, ProfileSVG
using ChainRules
# ccall((:openblas_get_num_threads64_, Base.libblas_name), Cint, ())

m, dss = deserialize("threadtest/model_and_samples.jls")
ds = catobs(dss...);
m(ds)
ps = Flux.params(m);
@btime m(ds)								#
@btime m(ds)								#  
@btime gradient(() -> sum(m(ds).data), ps)  #  
@btime gradient(() -> sum(m(ds).data), ps)  #


# Debugging slow matmul
B = ds[:importTable][:imports].data[:libraryName]
m = ArrayModel(Chain(Dense(2053,32,relu), Dense(32,32)))
ps = Flux.params(m)
Profile.clear()
@profile gradient(() -> sum(m(B).data), ps)
ProfileSVG.save("/tmp/profile.svg")

W = randn(Float32,64,64)
X = randn(Float32,64,640000)
Profile.clear()
@profile gradient(W -> sum(W * X), W)
ProfileSVG.save("/tmp/profile.svg")


######
# Non-threaded version on single thread
######
using Mill, BenchmarkTools, Serialization, Zygote, Flux
using LinearAlgebra
using Setfield
using ThreadPools
m, dss = deserialize("model_and_samples.jls")
ds = catobs(dss...);
ps = Flux.params(m);
@btime m(ds)								#
@btime m(ds)								# 46.100 ms (3385 allocations: 27.02 MiB)
@btime gradient(() -> sum(m(ds).data), ps)  #  
@btime gradient(() -> sum(m(ds).data), ps)  # 130.273 ms (14561 allocations: 82.94 MiB)


dss = [ds[i] for i in 1:Mill.nobs(ds)];
#time of the inference
@btime m(ds)
@btime ThreadPools.qmap(oneds -> m(oneds), dss)
@btime ThreadPools.qmap(oneds -> gradient(() -> sum(m(dss).data), ps), dss)

