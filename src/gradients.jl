using MacroTools: splitdef

# using FiniteDifferences on a struct instead of the result of Flux.params(m) leads to painfully
# slow computation, we need to stay with implicit parametrization for numeric gradcheck
numgrad(m, f, xs...) = FiniteDifferences.grad(m, f, xs...)
function numgrad(m, f, ps::Flux.Params)
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
    bg = numgrad(backward_fdm(5, 1), f, xs...)
    cg = numgrad(central_fdm(5, 1), f, xs...)
    fg = numgrad(forward_fdm(5, 1), f, xs...)
    zip(bg, cg, fg) |> collect
end

gradcomp(ag1::Nothing, ag2::Nothing, args...) = true
gradcomp(ag, ng, atol, rtol) = isapprox(ag, ng; atol, rtol)
gradcomp(::Nothing, ng, atol, rtol) = isapprox(ng, 0; atol, rtol)
gradcomp(ag, ngs::Tuple, atol, rtol) = mapreduce(ng -> isapprox.(ag, ng; atol, rtol), .|, ngs) |> all
gradcomp(::Nothing, ngs::Tuple, atol, rtol) = mapreduce(ng -> isapprox.(ng, 0; atol, rtol), .|, ngs) |> all
gradcomp(ag::NotImplemented, ngs, atol, rtol) = NotImplementedException(ag) |> throw
gradcomp(ag::NotImplemented, ngs::Tuple, atol, rtol) = NotImplementedException(ag) |> throw

# it is ok to use Float32 weights even for Float64 versions
gradf(f::Function, xs...) = gradf(rand(Float32, size(f(xs...))...), f)
gradf(f::Function, ::Flux.Params) = gradf(rand(Float32, size(f())...), f)
gradf(A, f::Function) = A, (xs...) -> sum(A .* f(xs...))

# extend f32 and f64 definitions from Flux to all numbers for gradient testing
mf32(x::Number) = Float32(x)
mf32(x) = Flux.f32(x)
mf64(x::Number) = Float64(x)
mf64(x) = Flux.f64(x)

# compute numerical gradient with double precision and compare to analytical gradient computed in single precision
# also check analytical gradient in either precisions against each other to see whether the implementation
# suffers from insufficient numerical precision
macro gradtest(gf, cvars=:[], atol=1e-5, rtol=1e-5)
    args = splitdef(gf)[:args]
    @assert cvars isa Expr && cvars.head == :vect "@gradtest accepts an array of closured variables as a second argument"
    cvars = cvars.args
    args32 = Symbol.(string.(args) .* "32")
    args64 = Symbol.(string.(args) .* "64")
    cvars32 = Symbol.(string.(cvars) .* "32")
    cvars64 = Symbol.(string.(cvars) .* "64")
    asgn32 = [:( $v = mf32($(esc(x))) ) for (v, x) in zip([args32; cvars32], [args; cvars])]
    asgn64 = [:( $v = mf64($(esc(x))) ) for (v, x) in zip([args64; cvars64], [args; cvars])]
    gf32 = MacroTools.postwalk(e -> e ∈ cvars ? Symbol("$(e)32") : e isa Symbol ? esc(e) : e, gf)
    gf64 = MacroTools.postwalk(e -> e ∈ cvars ? Symbol("$(e)64") : e isa Symbol ? esc(e) : e, gf)
    quote
        $(asgn32...)
        $(asgn64...)
        A, gf32 = gradf($gf32, $(args32...))
        _, gf64 = gradf(A, $gf64)
        ag32 = Flux.gradient(gf32, $(args32...))
        ag64 = Flux.gradient(gf64, $(args64...))
        ngs = numgrads(gf64, $(args64...))
        all(zip(ag32, ag64, ngs)) do (x, y, z)
            gradcomp(x, y, $atol, $rtol) && gradcomp(x, z, $atol, $rtol)
        end
    end
end

macro pgradtest(gf, cvars=:[], atol=1e-5, rtol=1e-5)
    m = splitdef(gf)[:args]
    @assert length(m) == 1 "@pgradtest accepts a function with only one argument"
    m = only(m)
    @assert cvars isa Expr && cvars.head == :vect "@pgradtest accepts an array of closured variables as a second argument"
    cvars = cvars.args
    cvars32 = Symbol.(string.(cvars) .* "32")
    cvars64 = Symbol.(string.(cvars) .* "64")
    asgn32 = [:( $v = mf32($(esc(x))) ) for (v, x) in zip(cvars32, cvars)]
    asgn64 = [:( $v = mf64($(esc(x))) ) for (v, x) in zip(cvars64, cvars)]
    gf_body = splitdef(gf)[:body]
    gf32 = MacroTools.postwalk(gf_body) do e
        e == m ? :m32 : e ∈ cvars ? Symbol("$(e)32") : e isa Symbol ? esc(e) : e
    end
    gf64 = MacroTools.postwalk(gf_body) do e
        e == m ? :m64 : e ∈ cvars ? Symbol("$(e)64") : e isa Symbol ? esc(e) : e
    end
    quote
        $(asgn32...)
        $(asgn64...)
        m32 = Flux.f32($(esc(m)))
        m64 = Flux.f64($(esc(m)))
        ps32 = Flux.params(m32)
        ps64 = Flux.params(m64)
        A, gf32 = gradf(() -> $gf32, ps32)
        _, gf64 = gradf(A, () -> $gf64)
        ag32 = Flux.gradient(gf32, ps32)
        ag64 = Flux.gradient(gf64, ps64)
        ngs = numgrads(gf64, ps64)
        all(1:length(ps32)) do i
            gradcomp(ag32[ps32[i]], ag64[ps64[i]], $atol, $rtol) && gradcomp(ag32[ps32[i]], ngs[i], $atol, $rtol)
        end
    end
end
