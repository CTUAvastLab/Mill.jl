using Test
using Mill
using Mill: AlignedBags, Maybe
using Mill: segmented_sum_forw, segmented_mean_forw, segmented_max_forw
using Mill: segmented_pnorm_forw, segmented_lse_forw
using Mill: bagconv, convsum
using Mill: p_map, r_map
import Mooncake
import Mooncake.TestUtils
import Zygote
using Random: Xoshiro

# ---------------------------------------------------------------------------
# Helper: compare Zygote and Mooncake gradients for a scalar-valued function.
#   f    – function returning a matrix (or scalar)
#   args – differentiable arguments
# A random coefficient matrix collapses the matrix output to a scalar.
# ---------------------------------------------------------------------------
function compare_grads(f, args...; atol=1f-4, rtol=1f-4)
    y_ref = f(args...)
    A = randn(eltype(y_ref), size(y_ref))
    g = (args...) -> sum(A .* f(args...))

    # Zygote reference gradients.
    gz = Zygote.gradient(g, args...)

    # Mooncake gradients.
    cache = Mooncake.prepare_gradient_cache(
        g, args...; config=Mooncake.Config(; friendly_tangents=true)
    )
    _, gm_tuple = Mooncake.value_and_gradient!!(cache, g, args...)
    # value_and_gradient!! returns (val, (grad_g, grad_arg1, ...)); drop grad_g.
    gm = Base.tail(gm_tuple)

    for (gz_i, gm_i) in zip(gz, gm)
        gz_i === nothing && continue
        @test gz_i ≈ gm_i atol=atol rtol=rtol broken=false
    end
end

@testset "Mooncake extension" begin

    T = Float32
    d, n = 4, 6

    x    = randn(T, d, n)
    ψ    = randn(T, d)
    ρ    = randn(T, d)
    bags = AlignedBags([1:2, 3:4, 5:6])
    rng  = Xoshiro(42)

    # Only test reverse mode: we have rrule!! but not frule!!.
    rev = Mooncake.ReverseMode

    @testset "segmented_sum (matrix x)" begin
        compare_grads((x, ψ) -> segmented_sum_forw(x, ψ, bags, nothing), x, ψ)
        TestUtils.test_rule(rng, segmented_sum_forw, x, ψ, bags, nothing;
            is_primitive=true, mode=rev)
    end

    @testset "segmented_sum (missing x)" begin
        compare_grads(ψ -> segmented_sum_forw(missing, ψ, bags, nothing), ψ)
        TestUtils.test_rule(rng, segmented_sum_forw, missing, ψ, bags, nothing;
            is_primitive=true, mode=rev)
    end

    @testset "segmented_mean (matrix x)" begin
        compare_grads((x, ψ) -> segmented_mean_forw(x, ψ, bags, nothing), x, ψ)
        TestUtils.test_rule(rng, segmented_mean_forw, x, ψ, bags, nothing;
            is_primitive=true, mode=rev)
    end

    @testset "segmented_mean (missing x)" begin
        compare_grads(ψ -> segmented_mean_forw(missing, ψ, bags, nothing), ψ)
        TestUtils.test_rule(rng, segmented_mean_forw, missing, ψ, bags, nothing;
            is_primitive=true, mode=rev)
    end

    @testset "segmented_max (matrix x)" begin
        compare_grads((x, ψ) -> segmented_max_forw(x, ψ, bags), x, ψ)
        TestUtils.test_rule(rng, segmented_max_forw, x, ψ, bags;
            is_primitive=true, mode=rev)
    end

    @testset "segmented_max (missing x)" begin
        compare_grads(ψ -> segmented_max_forw(missing, ψ, bags), ψ)
        TestUtils.test_rule(rng, segmented_max_forw, missing, ψ, bags;
            is_primitive=true, mode=rev)
    end

    @testset "segmented_pnorm (matrix a)" begin
        a = randn(T, d, n)
        p = p_map.(ρ)   # p > 1
        compare_grads((a, ψ, p) -> segmented_pnorm_forw(a, ψ, p, bags, nothing), a, ψ, p)
        # cap FD step to stay inside the domain (p > 1, |a| > 0)
        TestUtils.test_rule(rng, segmented_pnorm_forw, a, ψ, p, bags, nothing;
            is_primitive=true, mode=rev, max_fd_step=1e-3)
    end

    @testset "segmented_pnorm (missing a)" begin
        compare_grads(ψ -> segmented_pnorm_forw(missing, ψ, nothing, bags, nothing), ψ)
        TestUtils.test_rule(rng, segmented_pnorm_forw, missing, ψ, nothing, bags, nothing;
            is_primitive=true, mode=rev)
    end

    @testset "segmented_lse (matrix x)" begin
        r = r_map.(ρ)   # r > 0
        compare_grads((x, ψ, r) -> segmented_lse_forw(x, ψ, r, bags), x, ψ, r)
        # cap FD step to stay inside the domain (r > 0)
        TestUtils.test_rule(rng, segmented_lse_forw, x, ψ, r, bags;
            is_primitive=true, mode=rev, max_fd_step=1e-3)
    end

    @testset "segmented_lse (missing x)" begin
        r = r_map.(ρ)
        compare_grads(ψ -> segmented_lse_forw(missing, ψ, r, bags), ψ)
        TestUtils.test_rule(rng, segmented_lse_forw, missing, ψ, r, bags;
            is_primitive=true, mode=rev)
    end

    @testset "bagconv" begin
        W1 = randn(T, d, d)
        W2 = randn(T, d, d)
        compare_grads((x, W1, W2) -> bagconv(x, bags, W1, W2), x, W1, W2)
        TestUtils.test_rule(rng, bagconv, x, bags, W1, W2;
            is_primitive=true, mode=rev)
    end

    @testset "convsum" begin
        x1, x2, x3 = randn(T, d, n), randn(T, d, n), randn(T, d, n)
        compare_grads((x1, x2, x3) -> convsum(bags, x1, x2, x3), x1, x2, x3)
        TestUtils.test_rule(rng, convsum, bags, x1, x2, x3;
            is_primitive=true, mode=rev)
    end

    @testset "PreImputingMatrix (_mul_pi_maybe)" begin
        B = Matrix{Maybe{T}}(randn(T, d, n))
        B[1, 1] = missing
        B[2, 3] = missing
        compare_grads(ψ -> Mill._mul_pi_maybe(ψ, B), ψ)
        TestUtils.test_rule(rng, Mill._mul_pi_maybe, ψ, B;
            is_primitive=true, mode=rev)
    end

    @testset "PostImputingMatrix (_mul_pi_maybe_hot)" begin
        idxs = Maybe{Int}[1, missing, 2, 1, missing, 2]
        B    = maybehotbatch(idxs, 1:d)
        W    = randn(T, d, d)
        ψ0   = copy(ψ)
        # compare_grads tests W and ψ separately: each gradient is a plain array,
        # directly comparable between Zygote (NamedTuple) and Mooncake (Tangent).
        compare_grads(W -> Mill._mul_pi_maybe_hot(PostImputingMatrix(W, ψ0), B), W)
        compare_grads(ψ -> Mill._mul_pi_maybe_hot(PostImputingMatrix(copy(W), ψ), B), ψ0)
        TestUtils.test_rule(rng, Mill._mul_pi_maybe_hot, PostImputingMatrix(W, ψ0), B;
            is_primitive=true, mode=rev)
    end

    @testset "PostImputingMatrix (_mul_pi_ngram)" begin
        output_dim = 10
        seqs = Maybe{String}["ab", missing, "cd", "ab", missing, "ef"]
        B    = NGramMatrix(seqs, 3, 256, output_dim)
        W    = randn(T, d, output_dim)
        ψ0   = copy(ψ)
        compare_grads(W -> Mill._mul_pi_ngram(PostImputingMatrix(W, ψ0), B), W)
        compare_grads(ψ -> Mill._mul_pi_ngram(PostImputingMatrix(copy(W), ψ), B), ψ0)
        TestUtils.test_rule(rng, Mill._mul_pi_ngram, PostImputingMatrix(W, ψ0), B;
            is_primitive=true, mode=rev)
    end

end
