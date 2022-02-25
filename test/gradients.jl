# from https://github.com/jump-dev/JuMP.jl/blob/7aeec3fb2dc40c365b3f9c94d559fda5743d111f/test/utilities.jl#L41-L54

function _strip_line_from_error(err::ErrorException)
    return ErrorException(replace(err.msg, r"^At.+\:[0-9]+\: `@" => "In `@"))
end
_strip_line_from_error(err::LoadError) = _strip_line_from_error(err.error)
_strip_line_from_error(err) = err

# Test that the macro call `m` throws an error exception during pre-compilation
macro test_macro_throws(errortype, m)
    # See https://discourse.julialang.org/t/test-throws-with-macros-after-pr-23533/5878
    quote
        @test_throws(
            $(esc(_strip_line_from_error(errortype))),
            try
                @eval $m
            catch err
                throw(_strip_line_from_error(err))
            end
        )
    end
end

@testset "errors" begin
    @test_macro_throws AssertionError @gradtest 1
    @test_macro_throws AssertionError @gradtest "2"
    @test_macro_throws AssertionError @gradtest x + y

    @test_macro_throws AssertionError @pgradtest (m, ds) -> m(ds)
    @test_macro_throws AssertionError @pgradtest () -> m(ds)

    @test_macro_throws AssertionError @gradtest x -> x + y y
    @test_macro_throws AssertionError @pgradtest () -> m(x) x
end

@testset "array gradients" begin
    b = 1.0
    for a in [1.0, 2.0]
        @test @gradtest a -> 2a + 1 + sin(a)
        # all parameters for gradient computation use same precision
        @test @gradtest (a, b) -> begin
            typeof(a) == typeof(b) && a isa AbstractFloat || error()
            return 2a + b + 1 + sin(a * b)
        end
        # the same applies to explicitly stated closured variables
        @test @gradtest a -> begin
            typeof(a) == typeof(b) && a isa AbstractFloat || error()
            2a + b + 1 + sin(a * b)
        end [b]
        # if the variable is not explicitly stated as closured it retains its type
        @test @gradtest a -> begin
            typeof(b) == Float64 && a isa AbstractFloat || error()
            2a + b + 1 + sin(a * b)
        end
    end
    for a in [randn(2), randn(2, 2)]
        @test @gradtest a -> 2a .+ 1 .+ sin.(a)
        # all parameters for gradient computation use same precision
        @test @gradtest (a, b) -> begin
            eltype(a) == typeof(b) && a isa Array{<:AbstractFloat} && b isa AbstractFloat || error()
            return 2a .+ b .+ 1 .+ sin.(a .* b)
        end
        # the same applies to closured variables
        @test @gradtest a -> begin
            eltype(a) == typeof(b) && a isa Array{<:AbstractFloat} && b isa AbstractFloat || error()
            2a .+ b .+ 1 .+ sin.(a .* b)
        end [b]
        # if the variable is not explicitly stated as closured it retains its type
        @test @gradtest a -> begin
            typeof(b) == Float64 && a isa Array{<:AbstractFloat} || error()
            2a .+ b .+ 1 .+ sin.(a .* b)
        end
    end
end

@testset "parameter gradients" begin
    m = Dense(2, 2)
    b = randn(2, 2)
    for x in [randn(2), randn(2, 2)]
        @test @pgradtest m -> m(x)
        # closured variables
        @test @pgradtest m -> begin
            eltype(m.weight) == eltype(m.bias) == eltype(x) || error()
            m(x)
        end [x]
        @test @pgradtest m -> begin
            eltype(x) == Float64 || error()
            eltype(m.weight) == eltype(m.bias) && eltype(m.weight) <: AbstractFloat || error()
            m(x)
        end 
    end
end

@testset "gradf" begin
    gf = a -> 2a + 1 + sin(a)
    A, f = gradf(gf, 1)
    for i in [1,2,3]
        @test A*gf(i) == f(i)
    end
    # outputs are not constant
    @test f(1) ≠ f(2)
    @test f(2) ≠ f(3)
    # gradients do differ
    @test gradient(f, 1) ≠ gradient(f, 2)
    @test gradient(f, 2) ≠ gradient(f, 3)
    # 2nd derivative
    _, df = gradf(a -> gradient(f, a)[1], 1)
    @test df(1) ≠ df(2)
    @test df(2) ≠ df(3)
    @test gradient(df, 1) ≠ gradient(df, 2)
    @test gradient(df, 2) ≠ gradient(df, 3)

    A1 = randn(2, 2)
    A2 = randn(2, 2)
    gf = A -> 2A .+ 1 .+ sin.(A)
    A, f = gradf(gf, A1)
    @test f(A1) ≠ f(A2)
    @test sum(A .* gf(A1)) == f(A1)
    @test sum(A .* gf(A2)) == f(A2)
    @test gradient(f, A1) ≠ gradient(f, A2)
    _, df = gradf(A -> gradient(f, A)[1], A1)
    @test df(A1) ≠ df(A2)
    @test gradient(df, A1) ≠ gradient(df, A2)

    m1 = Dense(2, 2, relu)
    m2 = Dense(2, 2, relu)
    x = randn(2, 10)
    gf1 = () -> m1(x)
    gf2 = () -> m2(x)
    A1, f1 = gradf(gf1, Flux.params(m1))
    A2, f2 = gradf(gf2, Flux.params(m2))
    @test f1() ≠ f2()
    @test sum(A1 .* gf1()) == f1()
    @test sum(A2 .* gf2()) == f2()
    @test gradient(f1, Flux.params(m1)) ≠ gradient(f2, Flux.params(m2))
    @test gradient(f1, Flux.params(m1)) ≠ gradient(f1, Flux.params(m2))
    @test gradient(f2, Flux.params(m2)) ≠ gradient(f2, Flux.params(m1))
end

@testset "not implemented" begin
    f(x, y) = x + y
    function ChainRulesCore.rrule(::typeof(f), x, y)
        x+y, Δ -> (NoTangent(), Δ * 1, @not_implemented("Not implemented"))
    end
    x = y = 1.0
    @test_throws NotImplementedException @gradtest (x, y) -> f(x, y)
    @test_throws NotImplementedException @gradtest y -> f(1, y)
    @test @gradtest x -> f(x, 1)
end
