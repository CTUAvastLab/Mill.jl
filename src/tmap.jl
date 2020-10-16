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
