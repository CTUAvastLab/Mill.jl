@testset "aggregation grad check w.r.t. input" begin
    for bags in BAGS2
        d = rand(1:20)
        x = randn(d, 10)
        w = abs.(randn(size(x, 2))) .+ 0.1
        w_mat = abs.(randn(size(x))) .+ 0.1

        # generate all combinations of aggregations
        anames = ["Sum", "Mean", "Max", "PNorm", "LSE"]
        for idxs in powerset(collect(1:length(anames)))
            !isempty(idxs) || continue
            # not a thorough testing of all functions, but fast enough
            length(idxs) <= 3 || continue
            # for idxs in permutations(idxs)

            s = Symbol("Segmented", anames[idxs]...)
            a = @eval $s($d) |> f64
            @test mgradtest(x) do x
                a(x, bags)
            end
            @test mgradtest(x) do x
                a(x, bags, w)
            end
            @test mgradtest(x) do x
                a(x, bags, w_mat)
            end
        end
    end
end

@testset "aggregation grad check w.r.t. agg params" begin
    # r_map and p_map are stable
    @test first(gradient(softplus, 10000)) ≈ σ(10000) ≈ 1.0
    @test first(gradient(softplus, -10000)) ≈ σ(-10000) ≈ 0
     
    fs = [:SegmentedSum, :SegmentedMean, :SegmentedMax, :SegmentedPNorm, :SegmentedLSE]
    params = [(:ψ1,), (:ψ2,), (:ψ3,), (:ρ1, :c, :ψ4), (:ρ2, :ψ5)]

    for idxs in powerset(collect(1:length(fs)))
        !isempty(idxs) || continue;
        length(idxs) <= 2 || continue

        d = rand(1:20)
        x = randn(d, 0)
        as = []; cs = []; rs = []
        for (f, ps) in zip(fs[idxs], params[idxs])
            push!(rs, fill(:(randn($d)), length(ps))...)
            push!(as, ps...)
            push!(cs, Expr(:call, f, ps...))
        end
        @eval begin
            @test mgradtest($(map(eval, rs)...)) do $(as...)
                a = Aggregation($(cs...))
                a(missing, ScatteredBags([Int[], Int[]]))
            end
            @test mgradtest($(map(eval, rs)...)) do $(as...)
                a = Aggregation($(cs...))
                a(missing, AlignedBags([0:-1]), nothing)
            end
            @test mgradtest($(map(eval, rs)...)) do $(as...)
                a = Aggregation($(cs...))
                a($x, ScatteredBags([Int[]]))
            end
            @test mgradtest($(map(eval, rs)...)) do $(as...)
                a = Aggregation($(cs...))
                a($x, AlignedBags([0:-1, 0:-1]), nothing)
            end
        end

        for bags in BAGS2
            d = rand(1:20)
            x = randn(d, 10)
            w = abs.(randn(size(x, 2))) .+ 0.1
            w_mat = abs.(randn(size(x))) .+ 0.1
            as = []; cs = []; rs = []
            for (f, ps) in zip(fs[idxs], params[idxs])
                push!(rs, fill(:(randn($d)), length(ps))...)
                push!(as, ps...)
                push!(cs, Expr(:call, f, ps...))
            end
            @eval begin
                @test mgradtest($(map(eval, rs)...)) do $(as...)
                    a = Aggregation($(cs...))
                    a($x, $bags)
                end
                @test mgradtest($(map(eval, rs)...)) do $(as...)
                    a = Aggregation($(cs...))
                    a($x, $bags, $w)
                end
                @test mgradtest($(map(eval, rs)...)) do $(as...)
                    a = Aggregation($(cs...))
                    a($x, $bags, $w_mat)
                end
            end
        end
    end
end

@testset "derivative w.r.t weights in aggregations" begin
    for bags in BAGS2
        d = rand(1:20)
        x = randn(d, 10)
        w = abs.(randn(size(x, 2))) .+ 0.1
        w_mat = abs.(randn(size(x))) .+ 0.1

        a1 = SegmentedSum(d) |> f64
        a2 = SegmentedMean(d) |> f64
        a3 = SegmentedMax(d) |> f64
        a4 = SegmentedPNorm(d) |> f64
        a5 = SegmentedLSE(d) |> f64
        for g in [
                  w -> a1(x, bags, w),
                  w -> a2(x, bags, w),
                  w -> a3(x, bags, w),
                  w -> a5(x, bags, w)
                 ]
            @test mgradtest(g, w)
        end
        for g in [
                  w_mat -> a1(x, bags, w_mat),
                  w_mat -> a2(x, bags, w_mat),
                  w_mat -> a3(x, bags, w_mat),
                  w_mat -> a5(x, bags, w_mat)
                 ]
            @test mgradtest(g, w_mat)
        end
        # for g in [
        #           w -> a4(x, bags, w_mat)
        #          ]
        #     # NOT IMPLEMENTED YET
        #     @test_throws Exception mgradtest(g, w_mat)
        # end
        # for g in [
        #           w -> a4(x, bags, w_mat)
        #          ]
        #     # NOT IMPLEMENTED YET
        #     @test_throws Exception mgradtest(g, w_mat)
        # end
    end
end
