# Tests for CUDA extension
# Run these tests only when CUDA is available

using Mill
using Flux
using Test

# Only run tests if CUDA is functional
try
    using CUDA
    if CUDA.functional()
        @info "CUDA is available, running GPU tests"

        # The extension is automatically loaded when CUDA is imported
        # We access the CompressedBags through Base.get_extension
        MillCUDAExt = Base.get_extension(Mill, :MillCUDAExt)
        CompressedBags = MillCUDAExt.CompressedBags
        CuCompressedBags = MillCUDAExt.CuCompressedBags

        @testset "CompressedBags" begin
            # Test conversion from AlignedBags
            bags = Mill.AlignedBags([1:3, 4:5, 6:10])
            cbags = CompressedBags(bags)
            @test length(cbags) == 3
            @test cbags.num_observations == 10

            # Test GPU transfer
            gcbags = gpu(bags)
            @test gcbags isa CuCompressedBags

            # Test empty bags handling
            empty_bags = Mill.AlignedBags([1:2, 3:2, 4:5])  # middle bag is empty
            cempty = CompressedBags(empty_bags)
            @test length(cempty) == 3
        end

        @testset "ArrayNode GPU" begin
            x = ArrayNode(randn(Float32, 3, 5))
            gx = gpu(x)
            @test gx.data isa CuArray
            @test cpu(gx).data ≈ x.data
        end

        @testset "BagNode GPU" begin
            x = BagNode(ArrayNode(randn(Float32, 3, 10)), [1:3, 4:5, 6:10])
            gx = gpu(x)
            @test gx.data.data isa CuArray
            @test gx.bags isa MillCUDAExt.CuCompressedBags
        end

        @testset "ProductNode GPU" begin
            x = ProductNode((
                a = ArrayNode(randn(Float32, 3, 5)),
                b = ArrayNode(randn(Float32, 4, 5))
            ))
            gx = gpu(x)
            @test gx.data.a.data isa CuArray
            @test gx.data.b.data isa CuArray
        end

        @testset "SegmentedSum GPU" begin
            x = randn(Float32, 4, 10)
            bags = Mill.AlignedBags([1:3, 4:5, 6:10])
            ψ = zeros(Float32, 4)

            # CPU result
            agg = Mill.SegmentedSum(ψ)
            y_cpu = agg(x, bags, nothing)

            # GPU result
            gx = CuArray(x)
            gψ = CuArray(ψ)
            gbags = gpu(bags)
            y_gpu = Mill.segmented_sum_forw(gx, gψ, gbags, nothing)

            @test Array(y_gpu) ≈ y_cpu
        end

        @testset "SegmentedMean GPU" begin
            x = randn(Float32, 4, 10)
            bags = Mill.AlignedBags([1:3, 4:5, 6:10])
            ψ = zeros(Float32, 4)

            # CPU result
            agg = Mill.SegmentedMean(ψ)
            y_cpu = agg(x, bags, nothing)

            # GPU result
            gx = CuArray(x)
            gψ = CuArray(ψ)
            gbags = gpu(bags)
            y_gpu = Mill.segmented_mean_forw(gx, gψ, gbags, nothing)

            @test Array(y_gpu) ≈ y_cpu
        end

        @testset "SegmentedMax GPU" begin
            x = randn(Float32, 4, 10)
            bags = Mill.AlignedBags([1:3, 4:5, 6:10])
            ψ = zeros(Float32, 4)

            # CPU result
            agg = Mill.SegmentedMax(ψ)
            y_cpu = agg(x, bags, nothing)

            # GPU result
            gx = CuArray(x)
            gψ = CuArray(ψ)
            gbags = gpu(bags)
            y_gpu = Mill.segmented_max_forw(gx, gψ, gbags)

            @test Array(y_gpu) ≈ y_cpu
        end

        @testset "Full model GPU" begin
            # Create a simple BagNode structure
            ds = BagNode(ArrayNode(randn(Float32, 3, 10)), [1:3, 4:5, 6:10])

            # Create model
            model = reflectinmodel(ds, d -> Dense(d, 5), SegmentedMeanMax)

            # Move to GPU
            gds = gpu(ds)
            gmodel = gpu(model)

            # Forward pass on CPU
            y_cpu = model(ds)

            # Forward pass on GPU
            y_gpu = gmodel(gds)

            @test Array(y_gpu) ≈ y_cpu rtol=1e-4
        end

        @testset "Nested BagNode GPU" begin
            # Create nested structure
            inner = BagNode(ArrayNode(randn(Float32, 3, 20)), [1:4, 5:8, 9:12, 13:16, 17:20])
            outer = BagNode(inner, [1:2, 3:5])

            # Create and move model
            model = reflectinmodel(outer, d -> Dense(d, 5), SegmentedMean)

            gds = gpu(outer)
            gmodel = gpu(model)

            y_cpu = model(outer)
            y_gpu = gmodel(gds)

            @test Array(y_gpu) ≈ y_cpu rtol=1e-4
        end

        @testset "Gradient test GPU" begin
            ds = BagNode(ArrayNode(randn(Float32, 3, 10)), [1:3, 4:5, 6:10,0:-1])
            model = reflectinmodel(ds, d -> Dense(d, 5), SegmentedSumMeanMax)

            gds = gpu(ds)
            gmodel = gpu(model)

            # Test that gradients can be computed
            loss, grads = Flux.withgradient(gmodel) do m
                sum(m(gds))
            end

            @test !isnan(loss)
            @test grads !== nothing
        end

        @testset "Empty bags GPU" begin
            x = randn(Float32, 4, 5)
            bags = Mill.AlignedBags([1:2, 3:2, 4:5])  # middle bag is empty
            ψ = ones(Float32, 4)

            gx = CuArray(x)
            gψ = CuArray(ψ)
            gbags = gpu(bags)

            # SegmentedSum should fill empty bag with ψ
            y = Mill.segmented_sum_forw(gx, gψ, gbags, nothing)
            @test Array(y)[:, 2] ≈ ψ

            # SegmentedMean should fill empty bag with ψ
            y = Mill.segmented_mean_forw(gx, gψ, gbags, nothing)
            @test Array(y)[:, 2] ≈ ψ

            # SegmentedMax should fill empty bag with ψ
            y = Mill.segmented_max_forw(gx, gψ, gbags)
            @test Array(y)[:, 2] ≈ ψ
        end

        @testset "MaybeHotVector GPU" begin
            W = randn(Float32, 5, 10)
            gW = CuArray(W)

            # MaybeHotVector with integer index
            v1 = MaybeHotVector(UInt32(3), 10)
            @test gpu(v1) === v1  # Should stay unchanged
            @test cpu(v1) === v1

            # Multiplication: CuMatrix * MaybeHotVector{<:Integer}
            y_cpu = W * v1
            y_gpu = gW * v1
            @test Array(y_gpu) ≈ y_cpu

            # Test different indices
            for i in [1, 5, 10]
                v = MaybeHotVector(UInt32(i), 10)
                @test Array(gW * v) ≈ W * v
            end
        end

        @testset "MaybeHotMatrix GPU" begin
            W = randn(Float32, 5, 10)
            gW = CuArray(W)

            # MaybeHotMatrix with integer indices - moves to GPU (same type, different vector backend)
            M1 = MaybeHotMatrix(UInt32[1, 5, 3, 10], 10)
            gM1 = gpu(M1)
            @test gM1 isa MaybeHotMatrix{UInt32, <:CuVector{UInt32}, Bool}
            @test gM1.I isa CuVector
            @test cpu(gM1) == M1

            # Multiplication: CuMatrix * MaybeHotMatrix{..., CuVector, ...}
            y_cpu = W * M1
            y_gpu = gW * gM1
            @test Array(y_gpu) ≈ y_cpu

            # Also test CuMatrix * MaybeHotMatrix (CPU indices, auto-converts)
            y_gpu2 = gW * M1
            @test Array(y_gpu2) ≈ y_cpu

            # Test with different indices
            M2 = MaybeHotMatrix(UInt32[2, 2, 7], 10)
            @test Array(gW * gpu(M2)) ≈ W * M2
        end

        @testset "MaybeHotMatrix with missing GPU" begin
            W = randn(Float32, 5, 10)
            gW = CuArray(W)

            # All missing
            M_missing = MaybeHotMatrix(fill(missing, 3), 10)
            gM_missing = gpu(M_missing)
            @test gM_missing isa MaybeHotMatrix{Missing, <:CuVector{Missing}, Missing}
            y_gpu = gW * gM_missing
            @test size(y_gpu) == (5, 3)
            @test all(isnan, Array(y_gpu))  # NaN used as proxy for missing on GPU

            # Mixed case
            M_mixed = MaybeHotMatrix(Union{UInt32, Missing}[UInt32(2), missing, UInt32(5)], 10)
            gM_mixed = gpu(M_mixed)
            @test gM_mixed.I isa CuVector{Union{Missing, UInt32}}

            y_gpu = gW * gM_mixed
            y_cpu_approx = Array(y_gpu)
            @test y_cpu_approx[:, 1] ≈ W[:, 2]
            @test y_cpu_approx[:, 3] ≈ W[:, 5]
            # Column 2 has zeros (for missing)
            @test all(iszero, y_cpu_approx[:, 2])
        end

        @testset "PostImputingMatrix GPU" begin
            W = randn(Float32, 5, 10)
            ψ = randn(Float32, 5)
            A = PostImputingMatrix(W, ψ)
            gA = gpu(A)

            @test gA.W isa CuMatrix
            @test gA.ψ isa CuVector
            @test Array(gA.W) ≈ W
            @test Array(gA.ψ) ≈ ψ

            # Test cpu conversion
            A_back = cpu(gA)
            @test A_back.W ≈ W
            @test A_back.ψ ≈ ψ
        end

        @testset "PostImputingMatrix * MaybeHotVector GPU" begin
            W = randn(Float32, 5, 10)
            ψ = randn(Float32, 5)
            A = PostImputingMatrix(W, ψ)
            gA = gpu(A)

            # Integer index
            v1 = MaybeHotVector(UInt32(3), 10)
            y_cpu = A * v1
            y_gpu = gA * v1
            @test Array(y_gpu) ≈ y_cpu

            # Missing index - should return ψ
            v_missing = MaybeHotVector(missing, 10)
            y_cpu = A * v_missing
            y_gpu = gA * v_missing
            @test Array(y_gpu) ≈ ψ
        end

        @testset "PostImputingMatrix * MaybeHotMatrix GPU" begin
            W = randn(Float32, 5, 10)
            ψ = randn(Float32, 5)
            A = PostImputingMatrix(W, ψ)
            gA = gpu(A)

            # All integer indices
            M1 = MaybeHotMatrix(UInt32[1, 5, 3], 10)
            gM1 = gpu(M1)
            y_cpu = A * M1
            y_gpu = gA * gM1
            @test Array(y_gpu) ≈ y_cpu

            # Also test with CPU MaybeHotMatrix (auto-converts or uses CPU indices)
            y_gpu2 = gA * M1
            @test Array(y_gpu2) ≈ y_cpu

            # All missing
            M_missing = MaybeHotMatrix(fill(missing, 3), 10)
            gM_missing = gpu(M_missing)
            y_cpu = A * M_missing
            y_gpu = gA * gM_missing
            @test Array(y_gpu) ≈ y_cpu

            # Mixed case
            M_mixed = MaybeHotMatrix(Union{UInt32, Missing}[UInt32(2), missing, UInt32(5)], 10)
            gM_mixed = gpu(M_mixed)
            y_cpu = A * M_mixed
            y_gpu = gA * gM_mixed
            @test Array(y_gpu) ≈ y_cpu
        end

        @testset "PreImputingMatrix GPU" begin
            W = randn(Float32, 5, 10)
            ψ = randn(Float32, 10)
            A = PreImputingMatrix(W, ψ)
            gA = gpu(A)

            @test gA.W isa CuMatrix
            @test gA.ψ isa CuVector
            @test Array(gA.W) ≈ W
            @test Array(gA.ψ) ≈ ψ

            # Test cpu conversion
            A_back = cpu(gA)
            @test A_back.W ≈ W
            @test A_back.ψ ≈ ψ

            # Test regular multiplication (no missing)
            x = randn(Float32, 10, 4)
            gx = CuArray(x)
            y_cpu = A * x
            y_gpu = gA * gx
            @test Array(y_gpu) ≈ y_cpu
        end

        @testset "ArrayNode with MaybeHotMatrix GPU" begin
            # Test ArrayNode containing MaybeHotMatrix
            M = MaybeHotMatrix(UInt32[1, 3, 5, 2], 10)
            ds = ArrayNode(M)
            gds = gpu(ds)

            # MaybeHotMatrix indices should be on GPU
            @test gds.data isa MaybeHotMatrix{UInt32, <:CuVector, Bool}
            @test cpu(gds.data) == M

            # Create model and test forward pass
            model = ArrayModel(Dense(10, 5))
            gmodel = gpu(model)

            y_cpu = model(ds)
            y_gpu = gmodel(gds)
            @test Array(y_gpu) ≈ y_cpu rtol=1e-4
        end

        @testset "BagNode with MaybeHotMatrix GPU" begin
            # Test BagNode containing ArrayNode with MaybeHotMatrix
            M = MaybeHotMatrix(UInt32[1, 3, 5, 2, 7, 8], 10)
            inner = ArrayNode(M)
            ds = BagNode(inner, [1:2, 3:4, 5:6])

            gds = gpu(ds)
            @test gds.data.data isa MaybeHotMatrix{UInt32, <:CuVector, Bool}
            @test gds.bags isa MillCUDAExt.CuCompressedBags

            # Create model and test forward pass
            model = reflectinmodel(ds, d -> Dense(d, 5), SegmentedMean)
            gmodel = gpu(model)

            y_cpu = model(ds)
            y_gpu = gmodel(gds)
            @test Array(y_gpu) ≈ y_cpu rtol=1e-4
        end

        @testset "Full model with PostImputingDense GPU" begin
            # Test a model using PostImputingDense with MaybeHotMatrix input
            M = MaybeHotMatrix(Union{UInt32, Missing}[UInt32(1), missing, UInt32(3), UInt32(2)], 10)
            ds = ArrayNode(M)

            # Create model with PostImputingDense
            model = ArrayModel(postimputing_dense(10, 5))
            gmodel = gpu(model)
            gds = gpu(ds)

            @test gds.data.I isa CuVector

            y_cpu = model(ds)
            y_gpu = gmodel(gds)
            @test Array(y_gpu) ≈ y_cpu rtol=1e-4
        end

        @testset "Gradient test with MaybeHotMatrix GPU" begin
            # Test gradient computation with MaybeHotMatrix data
            M = MaybeHotMatrix(UInt32[1, 3, 5, 2], 10)
            ds = ArrayNode(M)

            model = ArrayModel(Dense(10, 5))
            gmodel = gpu(model)
            gds = gpu(ds)

            @test gds.data.I isa CuVector

            # Test that gradients can be computed
            loss, grads = Flux.withgradient(gmodel) do m
                sum(m(gds))
            end

            @test !isnan(loss)
            @test grads !== nothing
            @test grads[1].m.weight !== nothing
        end

        @testset "Gradient test with PostImputingMatrix and MaybeHotMatrix GPU" begin
            # Test gradient computation with PostImputingMatrix and mixed MaybeHotMatrix
            M = MaybeHotMatrix(Union{UInt32, Missing}[UInt32(1), missing, UInt32(3)], 10)
            ds = ArrayNode(M)

            model = ArrayModel(postimputing_dense(10, 5))
            gmodel = gpu(model)
            gds = gpu(ds)

            @test gds.data.I isa CuVector

            # Test that gradients can be computed
            loss, grads = Flux.withgradient(gmodel) do m
                sum(m(gds))
            end

            @test !isnan(loss)
            @test grads !== nothing
        end

        @testset "BagModel with PostImputingDense and MaybeHotMatrix GPU" begin
            # Complex test: BagModel with PostImputingDense processing MaybeHotMatrix
            M = MaybeHotMatrix(Union{UInt32, Missing}[UInt32(1), missing, UInt32(3), UInt32(2), UInt32(5), missing], 10)
            inner = ArrayNode(M)
            ds = BagNode(inner, [1:2, 3:4, 5:6])

            # Create model with PostImputingDense as instance model
            im = ArrayModel(postimputing_dense(10, 8))
            agg = SegmentedMean(8)
            bm = Dense(8, 4)
            model = BagModel(im, agg, bm)

            gmodel = gpu(model)
            gds = gpu(ds)

            @test gds.data.data.I isa CuVector

            y_cpu = model(ds)
            y_gpu = gmodel(gds)
            @test Array(y_gpu) ≈ y_cpu rtol=1e-4

            # Test gradients
            loss, grads = Flux.withgradient(gmodel) do m
                sum(m(gds))
            end
            @test !isnan(loss)
            @test grads !== nothing
        end

    else
        @warn "CUDA not functional, skipping GPU tests"
    end
catch e
    if isa(e, ArgumentError) && occursin("Package CUDA", string(e))
        @info "CUDA not installed, skipping GPU tests"
    else
        rethrow(e)
    end
end


# Manual benchmarking function - requires CUDA and BenchmarkTools
# Run interactively with: using BenchmarkTools; performance_test()
#=
function performance_test()
    ds = map((1,2,3,4)) do _
        ls = rand(0:10, 1000)
        inner = BagNode(ArrayNode(randn(Float32, 64, sum(ls))), Mill.length2bags(ls))
    end |> ProductNode

    # Create and move model
    model = reflectinmodel(ds, d -> Dense(d, 32), SegmentedMeanMax)

    gds = gpu(ds)
    gmodel = gpu(model)
    y_cpu = model(ds)
    y_gpu = gmodel(gds)

    gradient(m -> sum(m(ds)), model)
    gradient(m -> sum(m(gds)), gmodel)

    @benchmark gradient(m -> sum(m(ds)), model)
    @benchmark CUDA.@sync gradient(m -> sum(m(gds)), gmodel)

    @test Array(y_gpu) ≈ y_cpu rtol=1e-4
end
=#
