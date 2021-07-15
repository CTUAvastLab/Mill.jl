@testset "numerical gradients" begin
    f = a -> 2a .+ 1 .+ sin.(a)
    @test gradtest(f, 1.0)
    @test gradtest(f, 2.0)
    @test gradtest(f, randn(2, 2))
    b = 4
    f = a -> 2a .+ b .+ 1 .+ sin.(a.*b)
    @test gradtest(f, 1.0)
    @test gradtest(f, 2.0)
    @test gradtest(f, randn(2, 2))

    m1 = Dense(2, 2, relu) |> f64
    m2 = Dense(2, 2, relu) |> f64
    x = randn(2, 10)
    @test gradtest(() -> m1(x), Flux.params(m1))
    @test gradtest(() -> m2(x), Flux.params(m2))
end

@testset "gradf" begin
    b = 4
    f = gradf(a -> 2a .+ b .+ 1 .+ sin.(a.*b), 1)
    # outputs are not constant
    @test f(1) ≠ f(2)
    @test f(2) ≠ f(3)
    # gradients do differ
    @test gradient(f, 1) ≠ gradient(f, 2)
    @test gradient(f, 2) ≠ gradient(f, 3)
    # 2nd derivative
    df = gradf(f, 1)
    @test df(1) ≠ df(2)
    @test df(2) ≠ df(3)
    @test gradient(df, 1) ≠ gradient(df, 2)
    @test gradient(df, 2) ≠ gradient(df, 3)

    B = randn(2, 2)
    A1 = randn(2, 2)
    A2 = randn(2, 2)
    f = gradf(A -> 2A .+ B .+ 1 .+ sin.(A.*B), A1)
    @test f(A1) ≠ f(A2)
    @test gradient(f, A1) ≠ gradient(f, A2)
    df = gradf(f, A1)
    @test df(A1) ≠ df(A2)
    @test gradient(df, A1) ≠ gradient(df, A2)

    m1 = Dense(2, 2, relu) |> f64
    m2 = Dense(2, 2, relu) |> f64
    x = randn(2, 10)
    f1 = gradf(() -> m1(x), Flux.params(m1))
    f2 = gradf(() -> m2(x), Flux.params(m2))
    @test f1() ≠ f2()
    @test gradient(f1, Flux.params(m1)) ≠ gradient(f2, Flux.params(m2))
    @test gradient(f1, Flux.params(m1)) ≠ gradient(f1, Flux.params(m2))
    @test gradient(f2, Flux.params(m2)) ≠ gradient(f2, Flux.params(m1))
    df1 = gradf(f1, Flux.params(m1))
    df2 = gradf(f2, Flux.params(m2))
    @test df1() ≠ df2()
end

@testset "not implemented" begin
    f(x, y) = x + y
    function ChainRulesCore.rrule(::typeof(f), x, y)
        x+y, Δ -> (NoTangent(), Δ * 1, @not_implemented("Not implemented"))
    end
    @test_throws NotImplementedException gradtest(f, 1.0, 1.0)
    @test_throws NotImplementedException gradtest(y -> f(1, y), 1.0)
    @test gradtest(x -> f(x, 1), 1.0)
end
