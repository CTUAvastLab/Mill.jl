using Mill, BenchmarkTools, Serialization, Zygote, Flux
using LinearAlgebra
using Setfield
using ThreadPools
# ccall((:openblas_get_num_threads64_, Base.libblas_name), Cint, ())

(m, ds) = deserialize("gvma_threadtest.jls")
ps = Flux.params(m);
@btime m(ds)								#
@btime m(ds)								#  
@btime gradient(() -> sum(m(ds).data), ps)  #  
@btime gradient(() -> sum(m(ds).data), ps)  #



######
# Non-threaded version on single thread
######
using Mill, BenchmarkTools, Serialization, Zygote, Flux
using LinearAlgebra
using Setfield
using ThreadPools
(m, ds) = deserialize("../gvma_threadtest.jls")
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
