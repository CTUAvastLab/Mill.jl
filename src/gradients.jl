using MacroTools: splitdef
using Test

#=
Gradient checking similar to what is done in Zygote, but with support for
    -`nothing` analytical gradients (comparing numerical gradients to zero)
    - automatic summing of output with random coefficients to break symmetry, and
    - besides comparison of analytical and numerical gradients, also comparison of
      float32 to float64 precision gradients to discover numerical instability
    - closured variables are converted to the same precision (32 or 64) to simulate real world use,
      this is why a macro is needed
=#

# FiniteDifferences works on a struct but leads to painfully slow computation. Also, the resulting
# struct is hard to compare to the result of `Flux.gradient`.

# for arrays and numbers call `FiniteDifferences.grad` directly
numgrads(m, f, x::Union{AbstractArray, Number}) = FiniteDifferences.grad(m, f, x)
# for structs, return a tree-like structure using `fmapstructure`
function numgrads(m, f, s)
    function gradwith(x::AbstractArray)
        x_copy = copy(x)
        g = FiniteDifferences.grad(m, y -> (x .= y; f(s)), x) |> only
        copyto!(x, x_copy)
        g
    end
    Flux.fmapstructure(gradwith, s,
        exclude=(x -> x isa AbstractArray && !(x isa ImputingMatrix))
    ) |> tuple
end
# for multiple arguments, compute each one separately
function numgrads(m, f, arg, args...)
    args = (arg, args...)
    grads = map(eachindex(args)) do i
        numgrads(m, x -> f(args[1:i-1]..., x, args[i+1:end]...), args[i])
    end
    tuple(map(only, grads)...)
end

# For points that are close to discontinuities or non-differentiable parts compute both directions
function numgrads_all(f, args...)
    fg = numgrads(forward_fdm(5, 1), f, args...)
    bg = numgrads(backward_fdm(5, 1), f, args...)
    zip(fg, bg)
end

gradcomp(g::NotImplemented, args...; kwargs...) = NotImplementedException(g) |> throw

# two analytical grads
gradcomp(g1, g2; kwargs...) = isapprox(g1, g2; kwargs...)
gradcomp(::Nothing, ::Nothing, args...; kwargs...) = true
gradcomp(::Nothing, g; kwargs...) = isapprox(g, 0; kwargs...)
function gradcomp(g1::NamedTuple{K}, g2::NamedTuple{K}; kwargs...) where K
    all(gradcomp(g1[k], g2[k]; kwargs...) for k in K)
end
function gradcomp(g1::Tuple, g2::Tuple; kwargs...)
    length(g1) == length(g2) && all(gs -> gradcomp(gs...; kwargs...), zip(g1, g2))
end

# analytical vs numerical grads
function gradcomp(ag::Real, fg::T, bg::T; kwargs...) where T <: Real
    isapprox(ag, fg; kwargs...) || isapprox(ag, bg; kwargs...)
end
function gradcomp(ag::AbstractArray, fg::T, bg::T; kwargs...) where T <: AbstractArray
    all(isapprox.(ag, fg; kwargs...) .| isapprox.(ag, bg; kwargs...))
end
gradcomp(::Nothing, fg::T, bg::T; kwargs...) where T <: AbstractArray = gradcomp(zero(fg), fg, bg; kwargs...)
gradcomp(::Nothing, fg::NTuple{0}, bg::NTuple{0}; kwargs...) = true
function gradcomp(::Nothing, fg::T, bg::T; kwargs...) where T <: NamedTuple{K} where K
    all(gradcomp(nothing, fg[k], bg[k]; kwargs...) for k in K)
end
function gradcomp(ag::NamedTuple{K}, fg::T, bg::T; kwargs...) where T <: NamedTuple{K} where K
    all(gradcomp(ag[k], fg[k], bg[k]; kwargs...) for k in K)
end
function gradcomp(ag::Tuple, fg::Tuple, bg::Tuple; kwargs...)
    length(ag) == length(fg) == length(bg) && all(gs -> gradcomp(gs...; kwargs...), zip(ag, fg, bg))
end

# it is ok to use Float32 weights even for Float64 versions
gradf(f::Function, xs...) = gradf(rand(Float32, size(f(xs...))...), f)
gradf(A, f::Function) = A, (xs...) -> sum(A .* f(xs...))

# extend f32 and f64 definitions from Flux to all numbers for gradient testing
mf32(x::Number) = Float32(x)
mf32(x) = Flux.f32(x)
mf64(x::Number) = Float64(x)
mf64(x) = Flux.f64(x)

# we need a macro because of closured variables, which we convert to the same precision as the input
# to avoid Flux warnings.
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
        @test all(zip(
            Flux.gradient(gf32, $(args32...)),
            Flux.gradient(gf64, $(args64...)),
            numgrads_all(gf64, $(args64...))
        )) do (ag32, ag64, (fg, bg))
            gradcomp(ag32, ag64; atol=$atol, rtol=$rtol) && gradcomp(ag32, fg, bg; atol=$atol, rtol=$rtol)
        end
    end
end
