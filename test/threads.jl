using Mill, BenchmarkTools, Serialization, ThreadsX, Zygote, Flux
(m, ds) = deserialize("/tmp/testdata.jls")
dss = [ds[i] for i in 1:Mill.nobs(ds)]
m(ds)
#time of the inference
@btime m(ds)
@btime map(oneds -> m(oneds), dss)
#no threadding
# 588.425 ms (2387 allocations: 274.30 MiB)
#threadding inside Product Nodes
# 332.074 ms (4227 allocations: 274.40 MiB)
#threadding over samples
# 709.296 ms (96676 allocations: 279.28 MiB)

ps = Flux.params(m);
@btime gradient(() -> sum(m(ds).data), ps)


gradient(x -> sum(map(i -> 2i, x)), [1,2,3])[1]
gradient(x -> sum(ThreadsX.map(i -> 2i, x)), [1,2,3])[1]

gst = gradient(() -> sum(m(ds).data), ps)
# with intra sample threadding 
# julia> @btime gradient(() -> sum(m(ds).data), ps)
# 677.039 ms (20539 allocations: 765.46 MiB)

# without threadding
# julia> @btime gradient(() -> sum(m(ds).data), ps)
# 1.490 s (16625 allocations: 765.25 MiB



function ∇tmap(cx, f, args...)
    ys_and_backs = ThreadsX.map((args...) -> Zygote._pullback(cx, f, args...), args...)
    if isempty(ys_and_backs)
      ys_and_backs, _ -> nothing
    else
      ys, backs = Zygote.unzip(ys_and_backs)
      ys, function (Δ)
        # Apply pullbacks in reverse order. Needed for correctness if `f` is stateful.
        Δf_and_args_zipped = ThreadsX.map((f, δ) -> f(δ), Zygote._tryreverse(ThreadsX.map, backs, Δ)...)
        Δf_and_args = Zygote.unzip(Zygote._tryreverse(ThreadsX.map, Δf_and_args_zipped))
        Δf = reduce(Zygote.accum, Δf_and_args[1])
        (Δf, Δf_and_args[2:end]...)
      end
    end
end

Zygote.@adjoint function ThreadsX.map(f, args::Union{AbstractArray,Tuple}...)
    ∇tmap(__context__, f, args...)
end
