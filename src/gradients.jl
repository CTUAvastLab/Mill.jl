numgrad(m, f, xs...) = FiniteDifferences.grad(m, f, xs...)
function numgrad(m, f, ps::Flux.Params)
    ccp = deepcopy(ps)
    function gp(p)
        cp = deepcopy(p)
        g = FiniteDifferences.grad(m, x -> (p .= x; f()), p) |> only
        p .= cp 
        g
    end
    tuple(map(gp, ps)...)
end

# For points that are close to discontinuities or non-differentiable parts
function numgrads(f, xs...)
    bg = numgrad(backward_fdm(7, 1, max_range=1e-5), f, xs...)
    cg = numgrad(central_fdm(7, 1, max_range=1e-5), f, xs...)
    fg = numgrad(forward_fdm(7, 1, max_range=1e-5), f, xs...)
    zip(bg, cg, fg) |> collect
end

gradcheck(ag, ng; atol=1e-5, rtol=1e-5) = isnothing(ag) ? isapprox.(ng, 0; atol) : isapprox.(ng, ag; atol, rtol)
gradcheck(ag, ngs::Tuple; atol=1e-5, rtol=1e-5) = all(.|(map(ng -> gradcheck(ag, ng; atol, rtol), ngs)...))

function gradf(f, xs...)
    A = rand(size(f(xs...))...)
    (xs...) -> sum(A .* f(xs...))
end
function gradf(f, ps::Flux.Params)
    A = rand(size(f())...)
    () -> sum(A .* f())
end

function gradtest(f, xs...; atol=1e-5, rtol=1e-5)
    f2 = gradf(f, xs...)
    ag = Flux.gradient(f2, xs...)
    ngs = numgrads(f2, xs...)
    all(i -> gradcheck(ag[i], ngs[i]; atol, rtol), eachindex(xs))
end
function gradtest(f, ps::Flux.Params; atol=1e-5, rtol=1e-5)
    f2 = gradf(f, ps)
    ag = Flux.gradient(f2, ps)
    ngs = numgrads(f2, ps)
    all(i -> gradcheck(ag[ps[i]], ngs[i]; atol, rtol), 1:length(ps))
end
