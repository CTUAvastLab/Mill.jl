# The following code is taken from the maprreduce functionality of CuArrays.jl (with slight modifications to allow filtering based on bags)
# see https://github.com/JuliaGPU/CuArrays.jl/blob/master/src/mapreduce.jl
#
# Probably not the best in terms of efficiency, but still better than before.

using CuArrays: @cuindex, cudims
using CUDAnative
using CUDAdrv

function missingbags_mapreducedim_kernel_serial(f, op, R::CuDeviceArray{T}, A::CuDeviceArray{T}, bs, be, range) where T
    I = @cuindex R
    newrange = map((r, i) -> r === nothing ? i : r, range, I)
    for I′ in CartesianIndices(newrange)
        @inbounds Aval = be[I′[2]] < bs[I′[2]] ? A[I′] : zero(T)
        @inbounds R[I...] = op(R[I...], f(Aval))
    end
    return
end

function missingbags_mapreducedim_kernel_parallel(f, op, R::CuDeviceArray{T}, A::CuDeviceArray{T}, bs, be,
                             CIS, Rlength, Slength) where {T}
    for Ri_base in 0:(gridDim().x * blockDim().y):(Rlength-1)
        Ri = Ri_base + (blockIdx().x - 1) * blockDim().y + threadIdx().y
        Ri > Rlength && return
        RI = Tuple(CartesianIndices(R)[Ri])
        S = @cuStaticSharedMem(T, 512)
        Si_folded_base = (threadIdx().y - 1) * blockDim().x
        Si_folded = Si_folded_base + threadIdx().x
        # serial reduction of A into S by Slength ÷ xthreads
        for Si_base in 0:blockDim().x:(Slength-1)
            Si = Si_base + threadIdx().x
            Si > Slength && break
            SI = Tuple(CIS[Si])
            AI = ifelse.(size(R) .== 1, SI, RI)

            Aval = zero(T)
            if be[AI[2]] < bs[AI[2]]
                Aval = A[AI...]
            end

            if Si_base == 0
                S[Si_folded] = f(Aval)
            else
                S[Si_folded] = op(S[Si_folded], f(Aval))
            end
        end
        # block-parallel reduction of S to S[1] by xthreads
        CuArrays.reduce_block(view(S, (Si_folded_base + 1):512), op)
        # reduce S[1] into R
        threadIdx().x == 1 && (R[Ri] = op(R[Ri], S[Si_folded]))
    end
    return
end

function missingbags_mapreducedim!(f, op, R::CuArray{T}, A::CuArray{T}, bs, be) where {T}
    # the kernel as generated from `f` and `op` can require lots of registers (eg. #160),
    # so we need to be careful about how many threads we launch not to run out of them.
    Rlength = length(R)
    Ssize = ifelse.(size(R) .== 1, size(A), 1)
    Slength = prod(Ssize)
    CIS = CartesianIndices(Ssize)

    parallel_args = (f, op, R, A, bs, be, CIS, Rlength, Slength)
    GC.@preserve parallel_args begin
        parallel_kargs = cudaconvert.(parallel_args)
        parallel_tt = Tuple{Core.Typeof.(parallel_kargs)...}
        parallel_kernel = cufunction(missingbags_mapreducedim_kernel_parallel, parallel_tt)

        # we are limited in how many threads we can launch...
        ## by the kernel
        kernel_threads = CUDAnative.maxthreads(parallel_kernel)
        ## by the device
        dev = CUDAdrv.device()
        block_threads = (x=attribute(dev, CUDAdrv.DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X),
                         y=attribute(dev, CUDAdrv.DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y),
                         total=attribute(dev, CUDAdrv.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK))

        # figure out a legal launch configuration
        y_thr = min(nextpow(2, Rlength ÷ 512 + 1), 512, block_threads.y, kernel_threads)
        x_thr = min(512 ÷ y_thr, Slength, block_threads.x,
                    ceil(Int, block_threads.total/y_thr),
                    ceil(Int, kernel_threads/y_thr))

        if x_thr >= 8
            blk, thr = (Rlength - 1) ÷ y_thr + 1, (x_thr, y_thr, 1)
            parallel_kernel(parallel_kargs...; threads=thr, blocks=blk)
        else
            # not enough work, fall back to serial reduction
            range = ifelse.(length.(axes(R)) .== 1, axes(A), nothing)
            blk, thr = cudims(R)
            @cuda(blocks=blk, threads=thr, missingbags_mapreducedim_kernel_serial(f, op, R, A, bs, be, range))
        end
    end

    return R
end
