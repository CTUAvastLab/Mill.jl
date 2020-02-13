using CUDAnative, CuArrays, Mill
CuArrays.allowscalar(false)

Base.similar(::Nothing) = nothing
@inline weight(w::CuVector, ::Int, j::Int) = w[j]
@inline weight(w::CuMatrix, i::Int, j::Int) = w[i, j]

@inline bagnorm(w::Nothing, row,  start, stop) = stop - start + 1
@inline function bagnorm(w, row,  start, stop)
    o = weight(w, row, start)
    for i in start+1:stop
        o += weight(w, row, i)
    end
    o
end

isbagempty(bs, be, bi) = be[bi] < bs[bi]

function kernel_segmented_mean_forw!(y, x, C, bs, be, w)
    nrows = size(y,1)
    ri = threadIdx().x
    ri > nrows  && return(nothing)

    bi = blockIdx().x
    stride = blockDim().x
    @inbounds if isbagempty(bs, be, bi)
        for row in ri:stride:size(x,1)
            y[row, bi] = C[row]
        end
        return(nothing)
    end
    @inbounds for row in ri:stride:nrows
        acc = 0
        for j in bs[bi]:be[bi]
            ww = weight(w, row, j)
            # @cuprintln("thread $ri, block $stride, $j ww = $(ww)")
            acc +=  ww * x[row, j]
        end
        y[row, bi] = acc / bagnorm(w, row, bs[bi], be[bi])
    end
    return(nothing)
end

function kernel_missing_bags_back!(Δc, Δ, bs, be)
    bi = blockIdx().x
    nrows = length(Δc)
    ri = threadIdx().x
    ri > nrows  && return(nothing)
    stride = blockDim().x

    @inbounds for row in ri:stride:nrows
        acc = 0
        for i in 1:length(bs)
            !isbagempty(bs, be, i) && continue
            acc += Δ[row, i]
        end
        Δc[row] = acc
    end
    return(nothing)
end

function kernel_segmented_mean_back!(Δ, y, x, C, bs, be, w, Δx, Δc, Δw)
    bi = blockIdx().x
    bi > length(bs) && return(nothing)
    nrows = size(x,1)
    ri = threadIdx().x
    ri > nrows  && return(nothing)
    stride = blockDim().x
    isbagempty(bs, be, bi) && return(nothing)

    for row in ri:stride:nrows
        ws = bagnorm(w, row, bs[bi], be[bi])
        # @cuprintln("($ri, $bi) ws = $ws")
        Δrow = Δ[row, bi]
        for j in bs[bi]:be[bi]
            # @cuprintln("($ri, $bi) ($row, $j) ws = $(weight(w, row, j))")
            Δx[row, j] = weight(w, row, j) * Δrow / ws
            ∇dw_segmented_mean!(Δw, Δ, x, y, ws, row, j, bi)
        end
    end
    return(nothing)
end


function segmented_mean_forw(x::CuMatrix, c::CuVector, bags::CuAlignedBags, w)
    y = similar(x, size(x,1), length(bags))
    @cuda threads=256 blocks=length(bags) kernel_segmented_mean_forw!(y, x, c, bags.bs, bags.be, w)
    y
end

function segmented_mean_back(Δ::CuMatrix, y::CuMatrix, x::CuMatrix, c::CuVector, bags::CuAlignedBags, w::Union{Nothing, CuMatrix})
    Δx = similar(x)
    Δc = similar(c)
    Δw = similar(w)
    @cuda threads=256 blocks=length(bags) kernel_segmented_mean_back!(Δ, y, x, c, bags.bs, bags.be, w, Δx, Δc, Δw)
    @cuda threads=256 blocks=1 kernel_missing_bags_back!(Δc, Δ, bags.bs, bags.be)
    Δx, Δc, nothing, Δw
end

function segmented_mean_back(Δ::CuMatrix, y::CuMatrix, x::CuMatrix, c::CuVector, bags::CuAlignedBags, w::CuVector)
    Δx = similar(x)
    Δc = similar(c)
    Δw = similar(w, size(x))
    @cuda threads=256 blocks=length(bags) kernel_segmented_mean_back!(Δ, y, x, c, bags.bs, bags.be, w, Δx, Δc, Δw)
    @cuda threads=256 blocks=1 kernel_missing_bags_back!(Δc, Δ, bags.bs, bags.be)
    Δx, Δc, nothing, sum(Δw, dims = 1)[:]
end

function segmented_mean_back(Δ::CuMatrix, y::CuMatrix, x::Missing, C::CuVector, bags::CuAlignedBags, w::Nothing)
    # dC = zero(C)
    # @inbounds for (bi, b) in enumerate(bags)
    #     for i in eachindex(C)
    #         dC[i] += Δ[i, bi]
    #     end
    # end
    dC = sum(Δ, dims=2)
    nothing, dC[:], nothing, nothing
end

@adjoint function segmented_mean_forw(x::CuMatrix, c::CuVector, bags::CuAlignedBags, w)
    y = segmented_mean_forw(x, c, bags, w)
    y, Δ -> segmented_mean_back(Δ, y, x, c, bags, w)
end
