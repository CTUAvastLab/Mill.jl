using Pkg
Pkg.activate("..")
using CUDAnative, CuArrays, Mill
using Mill: ∇dw_segmented_mean!, segmented_mean_forw, segmented_mean_back
using Mill: weight, bagnorm
using Test
CuArrays.allowscalar(false)


Base.similar(::Nothing) = nothing
@inline Mill.weight(w::Nothing, _, _) = 1
@inline Mill.weight(w::CuVector, ::Int, j::Int) = w[j]
@inline Mill.weight(w::CuMatrix, i::Int, j::Int) = w[i, j]

@inline Mill.bagnorm(w::Nothing, row,  start, stop) = stop - start + 1
@inline function Mill.bagnorm(w, row,  start, stop)
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
    if isbagempty(bs, be, bi)
    	for row in ri:stride:size(x,1)
	        y[row, bi] = C[row]
	    end
        return(nothing)
    end
	for row in ri:stride:size(x,1)
	    y[row, bi] = 0
	    for j in bs[bi]:be[bi]
	    	ww = weight(w, row, j)
	    	# @cuprintln("thread $ri, block $stride, $j ww = $(ww)")
	        y[row, bi] +=  ww * x[row, j]
	    end
	    y[row, bi] /= bagnorm(w, row, bs[bi], be[bi])
	end
    return(nothing)
end

function kernel_missing_bags_back!(Δc, Δ, bs, be)
end

function kernel_segmented_mean_back!(Δ, y, x, C, bs, be, w, Δx, Δc, Δw) 
    bi = blockIdx().x
    nrows = size(x,1)
    ri = threadIdx().x
    ri > nrows  && return(nothing)
    stride = blockDim().x

    #we need to deal with empty bags here
    if bi == length(bs) + 1
    	for row in ri:stride:size(x,1)
    		Δc[row] = 0
	        for i in 1:length(bs)
	        	!isbagempty(bs, be, i) && continue
	            Δc[row] += Δ[row, i]
	        end
	    end
	    return(nothing)
    end
    isbagempty(bs, be, bi) && return(nothing)

    for row in ri:stride:nrows
        ws = bagnorm(w, row, bs[bi], be[bi])
	    # @cuprintln("($ri, $bi) ws = $ws")
        for j in bs[bi]:be[bi]
		    # @cuprintln("($ri, $bi) ($row, $j) ws = $(weight(w, row, j))")
	        Δx[row, j] = weight(w, row, j) * Δ[row, bi] / ws
            ∇dw_segmented_mean!(Δw, Δ, x, y, ws, row, j, bi)
	    end
    end
    return(nothing)
end


function Mill.segmented_mean_forw(x::CuMatrix, c::CuVector, bags::AlignedBags, w) 
	bs = CuArray([Int32(s.start) for s in bags])
	be = CuArray([Int32(s.stop) for s in bags])
	y = similar(x, size(x,1), length(bags))
    @cuda threads=256 blocks=length(bags) kernel_segmented_mean_forw!(y, x, c, bs, be, w)
    y
end

function Mill.segmented_mean_back(Δ::CuMatrix, y::CuMatrix, x::CuMatrix, c::CuVector, bags::AlignedBags, w::Union{Nothing, CuMatrix}) 
	bs = CuArray([Int32(s.start) for s in bags])
	be = CuArray([Int32(s.stop) for s in bags])
	Δx = similar(x)
	Δc = similar(c)
	Δw = similar(w)
    @cuda threads=256 blocks=length(bags) + 1 kernel_segmented_mean_back!(Δ, y, x, c, bs, be, w, Δx, Δc, Δw)
    Δx, Δc, nothing, Δw
end

function Mill.segmented_mean_back(Δ::CuMatrix, y::CuMatrix, x::CuMatrix, c::CuVector, bags::AlignedBags, w::CuVector) 
	bs = CuArray([Int32(s.start) for s in bags])
	be = CuArray([Int32(s.stop) for s in bags])
	Δx = similar(x)
	Δc = similar(c)
	Δw = similar(w, size(x))
    @cuda threads=256 blocks=length(bags) + 1 kernel_segmented_mean_back!(Δ, y, x, c, bs, be, w, Δx, Δc, Δw)
	Δx, Δc, nothing, sum(Δw, dims = 1)[:]
end

@testset "segmented mean GPU" begin 
	x = Matrix{Float64}(reshape(1:12, 2, 6))
	bags = AlignedBags([1:1, 2:3, 4:6, 0:-1])
	c = [1.0, -2]
	Δ = randn(2,4)
	gΔ = CuArray(Δ)

	gx = CuArray(x)
	gc = CuArray(c)
	for w in [nothing, abs.(randn(size(x,2))), abs.(randn(size(x)))] 
		gw = isnothing(w) ? nothing : CuArray(w)
		y = segmented_mean_forw(x, c, bags, w)
		gy = segmented_mean_forw(gx, gc, bags, gw)
		@test y ≈ gy

		Δx = segmented_mean_back(Δ, y, x, c, bags, w)[1]
		Δgx = segmented_mean_back(gΔ, gy, gx, gc, bags, gw)[1]
		@test Δx ≈ Δgx

		Δc = segmented_mean_back(Δ, y, x, c, bags, w)[2]
		Δgc = segmented_mean_back(gΔ, gy, gx, gc, bags, gw)[2]
		@test Δc ≈ Δgc

		Δw = segmented_mean_back(Δ, y, x, c, bags, w)[4]
		Δgw = segmented_mean_back(gΔ, gy, gx, gc, bags, gw)[4]
		!isnothing(w) && @test Δw ≈ Δgw
	end
end

# w = abs.(randn(size(x,2)))
# gw = CuArray(w)
# segmented_mean_forw(gx, gc, bags::AlignedBags, nothing) ≈ segmented_mean_forw(x, c, bags::AlignedBags, nothing) 
# w, gw = nothing, nothing
# segmented_mean_forw(gx, gc, bags::AlignedBags, nothing) ≈ segmented_mean_forw(x, c, bags::AlignedBags, nothing) 