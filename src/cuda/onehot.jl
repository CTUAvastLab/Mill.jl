using CuArrays, CUDAnative, CUDAdrv

"""
    Multiplication kernel for (A::CuMatrix) * (B::OneHotMatrix)
"""
function mul_onehot_kernel!(R::CuDeviceArray, A::CuDeviceArray, B::CuDeviceArray, m, n)
    i = (blockIdx().y-1) * blockDim().x + threadIdx().x
    j = (blockIdx().x-1) * blockDim().y + threadIdx().y

    i > m && return nothing
    j > n && return nothing

    @inbounds R[i,j] = A[i,B[j].ix]

    return nothing
end

"""
    Multiplies (A::CuMatrix) * (B::OneHotMatrix{<: CuArray})
"""
function Base.:*(A::CuMatrix, B::Flux.OneHotMatrix{<: CuArray})
    m = size(A,1)
    n = length(B.data)
    R = similar(A, m, n)
    parallel_args = (R, A, B.data, m, n)
    GC.@preserve parallel_args begin
        parallel_kargs = cudaconvert.(parallel_args)
        parallel_tt = Tuple{Core.Typeof.(parallel_kargs)...}
        parallel_kernel = cufunction(mul_onehot_kernel!, parallel_tt)

        kernel_threads = CUDAnative.maxthreads(parallel_kernel)
        dev = CUDAdrv.device()
        block_threads = (x=attribute(dev, CUDAdrv.DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X),
                         y=attribute(dev, CUDAdrv.DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y),
                         total=attribute(dev, CUDAdrv.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK))

        x_thr = min(512, block_threads.x, kernel_threads, block_threads.total, 32 * ((m ÷ 32) + 1))
        y_thr = min(512, block_threads.y, kernel_threads ÷ x_thr, block_threads.total ÷ x_thr, n)

        # BEWARE! The block indices are reversed due to larger limit on x-dim of grid
        y_blk = ceil(Int, m / x_thr)
        x_blk = ceil(Int, n / y_thr)

        parallel_kernel(parallel_kargs...; threads=(x_thr,y_thr), blocks=(x_blk,y_blk))
        synchronize()
    end

    return R
end

#TODO: Gradients w.r.t. matrix multiplication by one-hot matrices are not considered
Zygote.@adjoint function Base.:*(A::CuMatrix, B::Flux.OneHotMatrix{<: CuArray})
    A * B, Δ -> (nothing, nothing)
end
