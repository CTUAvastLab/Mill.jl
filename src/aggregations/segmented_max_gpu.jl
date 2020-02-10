
function kernel_segmented_max_forw!(o, maxI, x, C, bs, be) 
    nrows = size(o,1)
    ri = threadIdx().x
    ri > nrows  && return(nothing)

    bi = blockIdx().x
    stride = blockDim().x
    if isbagempty(bs, be, bi)
        @inbounds for row in ri:stride:size(x,1)
            o[row, bi] = C[row]
            maxI[row, bi] = 0
        end
        return(nothing)
    end
    @inbounds for row in ri:stride:size(x,1)
        o[row, bi] = x[row, bs[bi]]
        maxI[row, bi] = bs[bi]
        for j in bs[bi] + 1:be[bi]
            if x[row, j] > o[row, bi]
                o[row, bi] = x[row, j]
                maxI[row, bi] = j 
            end
        end
    end
    return(nothing)
end

function kernel_segmented_max_back!(Δ, maxI, x, bs, be, Δx, Δc) 
    bi = blockIdx().x
    nrows = size(x,1)
    ri = threadIdx().x
    ri > nrows  && return(nothing)
    stride = blockDim().x
    isbagempty(bs, be, bi) && return(nothing)

    @inbounds for row in ri:stride:nrows
        for j in bs[bi]:be[bi]
            Δx[row, j] = 0
        end
        Δx[row, maxI[row, bi]] = Δ[row, bi]
    end
    return(nothing)
end


function segmented_max_forw(x::CuMatrix, c::CuVector, bags::AlignedBags) 
    bs = CuArray([Int32(s.start) for s in bags])
    be = CuArray([Int32(s.stop) for s in bags])
    segmented_max_forw(x, c, bs, be) 
end

function segmented_max_forw(x::CuMatrix, c::CuVector, bs, be) 
    o = similar(x, size(x,1), length(bs))
    maxI = CuArrays.fill(0, size(x,1), length(bs))
    @cuda threads=256 blocks=length(bs) kernel_segmented_max_forw!(o, maxI, x, c, bs, be)
    o, maxI
end

function segmented_max_back(Δ::CuMatrix, maxI::CuMatrix, y::CuMatrix, x::CuMatrix, c::CuVector, bags::AlignedBags) 
    bs = CuArray([Int32(s.start) for s in bags])
    be = CuArray([Int32(s.stop) for s in bags])
    segmented_max_back(Δ, maxI, y, x, c, bs, be) 
end

function segmented_max_back(Δ::CuMatrix, maxI::CuMatrix, y::CuMatrix, x::CuMatrix, c::CuVector, bs, be) 
    Δx = similar(x)
    Δc = similar(c)
    @cuda threads=256 blocks=length(bs) kernel_segmented_max_back!(Δ, maxI, x, bs, be, Δx, Δc) 
    @cuda threads=256 blocks=1 kernel_missing_bags_back!(Δc, Δ, bs, be)
    Δx, Δc, nothing
end

@adjoint function segmented_max_forw(x::CuMatrix, c::CuVector, bags::AlignedBags)
    bs = CuArray([Int32(s.start) for s in bags])
    be = CuArray([Int32(s.stop) for s in bags])
    y, maxI = segmented_max_forw(x, c, bs, be)
    y, Δ -> segmented_max_back(Δ, maxI, y, x, c, bs, be) 
end

