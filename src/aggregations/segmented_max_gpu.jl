
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


function segmented_max_forw(x::CuMatrix, c::CuVector, bags::CuAlignedBags)
    segmented_max_forw(x, c, bags.bs, bags.be)
end

function segmented_max_forw_maxI(x::CuMatrix, c::CuVector, bs, be)
    o = similar(x, size(x,1), length(bs))
    maxI = CuArrays.zeros(Int32, size(x,1), length(bs))
    @cuda threads=256 blocks=length(bs) kernel_segmented_max_forw!(o, maxI, x, c, bs, be)
    synchronize()
    o, maxI
end

segmented_max_forw(x::CuMatrix, c::CuVector, bs, be) = segmented_max_forw_maxI(x, c, bs, be)[1]

#
# function segmented_max_forw(::Missing, C::CuVector, bags::CuAlignedBags)
#     out = CuArrays.zeros(length(C), length(bags))
#     out .= C
#     out
# end
#
# @adjoint function segmented_max_forw(::Missing, C::CuVector, bags::CuAlignedBags)
#     nothing, nothing, nothing
# end

function segmented_max_back(Δ::CuMatrix, maxI::CuMatrix, y::CuMatrix, x::CuMatrix, c::CuVector, bags::CuAlignedBags)
    segmented_max_back(Δ, maxI, y, x, c, bags.bs, bags.be)
end

function segmented_max_back(Δ::CuMatrix, maxI::CuMatrix, y::CuMatrix, x::CuMatrix, c::CuVector, bs, be)
    Δx = similar(x)
    Δc = similar(c)
    @cuda threads=256 blocks=length(bs) kernel_segmented_max_back!(Δ, maxI, x, bs, be, Δx, Δc)
    synchronize()
    @cuda threads=256 blocks=1 kernel_missing_bags_back!(Δc, Δ, bs, be)
    synchronize()
    Δx, Δc, nothing
end

function segmented_max_back(Δ, y, x::Missing, C, bags::CuAlignedBags)
    # dC = zero(C)
    # @inbounds for (bi, b) in enumerate(bags)
    #     for i in eachindex(C)
    #         dC[i] += Δ[i, bi]
    #     end
    # end
    # nothing, dC, nothing, nothing

    dC = sum(Δ, dims=2)
    nothing, dC[:], nothing, nothing
end

@adjoint function segmented_max_forw(x::Missing, c, bags::CuAlignedBags)
    y = repeat(c, 1, length(bags))
    y, Δ -> segmented_max_back(Δ, y, x, c, bags)
end

@adjoint function segmented_max_forw(x, c, bags::CuAlignedBags)
    y, maxI = segmented_max_forw_maxI(x, c, bags.bs, bags.be)
    y, Δ -> segmented_max_back(Δ, maxI, y, x, c, bags.bs, bags.be)
end
