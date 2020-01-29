using Flux
_convshift(n) = (i = div(n, 2); mod(n, 2) == 0 ? (1 - i:i) : (-i : i) )
"""
	_addmatvec!(o, i, W, x, j)

	add a product of matrix `W` with a j-th column of `x` to i-th columns of `o`

"""
function _addmatvec!(o::Matrix, i, W::Matrix, x::Matrix, j)
	@inbounds for s in 1:size(W, 2)
		for r in 1:size(W, 1)
			o[r, i] += W[r, s] * x[s, j]
		end
	end
end

function _addmatvec!(o::Matrix, i, W::Matrix, x::SparseMatrixCSC, j)
	rn = x.colptr[j]:x.colptr[j+1] - 1
	isempty(rn) && return
	rowptr = x.rowval[rn]
	vals = x.nzval[rn]
	@inbounds for (s, v) in zip(rowptr, vals)
		for r in 1:size(W, 1)
			o[r, i] += W[r, s] * v
		end
	end
end


"""
	_addmattvec!(o, i, W, x, j)

	add a product of a transposed matrix `W` with a j-th column of `x` to i-th columns of `o`

"""
function _addmattvec!(o::Matrix, i, W::Matrix, x::AbstractMatrix, j)
	@inbounds for s in 1:size(W, 1)
		for r in 1:size(W, 2)
			o[r, i] += W[s, r] * x[s, j]
		end
	end
end

"""
	_outeradd!(W, Δ, i, x, j)

	add an outer product of i-th column of `Δ` and transposed `j`-th columns of `x` to `W`

"""
function _addvecvect!(W::Matrix, Δ::Matrix, i, x::AbstractMatrix, j)
	@inbounds for r in 1:size(W, 2)
		for s in 1:size(W, 1)
			W[s, r] += Δ[s, i] * x[r, j]
		end
	end
end

function _addvecvect!(W::Matrix, Δ::Matrix, i, x::SparseMatrixCSC, j)
	rn = x.colptr[j]:x.colptr[j+1] - 1
	isempty(rn) && return
	rowptr = x.rowval[rn]
	vals = x.nzval[rn]
	@inbounds for (r, v) in zip(rowptr, vals)
		for s in 1:size(W, 1)
			W[s, r] += Δ[s, i] * v
		end
	end
end

function bagconv!(o, x, bags::T, W...) where {T<:Union{AlignedBags, Vector{UnitRange{Int64}}}}
	offsets = _convshift(length(W))
	for b in bags
		for ri in b 
			o[:, ri] .= 0
			for (i, k) in enumerate(offsets)
				if first(b) <= k + ri  <= last(b)
					_addmatvec!(o, ri, W[i], x, k + ri)
				end
			end
		end
	end
	o
end

function bagconv!(o, x, bags::T, W...) where {T<:Union{ScatteredBags,Vector{Vector{Int64}}}}
	offsets = _convshift(length(W))
	for b in bags
		for (bi, ri) in enumerate(b) 
			o[:, ri] .= 0
			for (i, k) in enumerate(offsets)
				if 0 < k + bi  <= length(b)
					_addmatvec!(o, ri, W[i], x, b[k+bi])
				end
			end
		end
	end
	o
end


function bagconv(x, bags, W...)
	o = similar(W[1], size(W[1], 1), size(x, 2))
	o .= 0
	for i in 1:Threads.nthreads()
		bagconv!(o, x, subsetof(bags), W...)
	end
	o
end

function ∇wbagconv!(∇W, Δ, x, bags::T, W...) where {T<:Union{AlignedBags, Vector{UnitRange{Int64}}}}
	offsets = _convshift(length(W))
	foreach(w -> w .= 0, ∇W)
	for b in bags
		for ri in b 
			for (i, k) in enumerate(offsets)
				if first(b) <= k + ri  <= last(b)
					_addvecvect!(∇W[i], Δ, ri, x, k + ri)
				end
			end
		end
	end
end

function ∇wbagconv!(∇W, Δ, x, bags::T, W...) where {T<:Union{ScatteredBags,Vector{Vector{Int64}}}}
	offsets = _convshift(length(W))
	foreach(w -> w .= 0, ∇W)
	for b in bags
		for (bi, ri) in enumerate(b) 
			for (i, k) in enumerate(offsets)
				if 0 < k + bi  <= length(b)
					_addvecvect!(∇W[i], Δ, ri, x, b[k+bi])
				end
			end
		end
	end
end

function ∇wbagconv(Δ, x, bags, W...)
	∇Ws = [[zero(w) for w in W] for i in 1:Threads.nthreads()]
	for i in 1:Threads.nthreads()
		∇wbagconv!(∇Ws[i], Δ, x, subsetof(bags), W...)
	end
	foreach(i -> foreach(x -> x[1] .+= x[2], zip(∇Ws[1], ∇Ws[i])), 2:Threads.nthreads())
	return(tuple(∇Ws[1]...))
end

function ∇xbagconv(Δ, x, bags::T, W...) where {T<:Union{AlignedBags, Vector{UnitRange{Int64}}}}
	offsets = _convshift(length(W))
	∇x = fill(zero(eltype(x)), size(x))
	for b in bags
		for ri in b 
			for (i, k) in enumerate(offsets)
				if first(b) <= k + ri  <= last(b)
					_addmattvec!(∇x, k + ri, W[i], Δ, ri)
				end
			end
		end
	end
	∇x
end

function ∇xbagconv(Δ, x, bags::T, W...) where {T<:Union{ScatteredBags,Vector{Vector{Int64}}}}
	offsets = _convshift(length(W))
	∇x = fill(zero(eltype(x)), size(x))
	for b in bags
		for (bi, ri) in enumerate(b) 
			for (i, k) in enumerate(offsets)
				if 0 < k + bi  <= length(b)
					_addmattvec!(∇x, b[k+bi], W[i], Δ, ri)
				end
			end
		end
	end
	∇x
end

function ∇xwbagconv(Δ, x, bags::T, W...) where {T<:Union{AlignedBags, Vector{UnitRange{Int64}}}}
	offsets = _convshift(length(W))
	∇x = fill(zero(eltype(x)), size(x))
	∇W = [zero(w) for w in W]
	for b in bags
		for ri in b 
			for (i, k) in enumerate(offsets)
				if first(b) <= k + ri  <= last(b)
					_addmattvec!(∇x, k + ri, W[i], Δ, ri)
					_addvecvect!(∇W[i], Δ, ri, x, k + ri)
				end
			end
		end
	end
	∇x, tuple(∇W...)
end

function ∇xwbagconv(Δ, x, bags::T, W...) where {T<:Union{ScatteredBags,Vector{Vector{Int64}}}}
	offsets = _convshift(length(W))
	∇x = zero(W[1])
	∇W = [zero(w) for w in W]
	for b in bags
		for (bi, ri) in enumerate(b) 
			for (i, k) in enumerate(offsets)
				if first(b) <= k + ri  <= last(b)
					_addmattvec!(∇x, b[k+bi], W[i], Δ, ri)
					_addvecvect!(∇W[i], Δ, ri, x, b[k+bi])
				end
			end
		end
	end
	∇x, tuple(∇W...)
end

Zygote.@adjoint function bagconv(x, bags, fs::Matrix...)
  bagconv(x, bags, fs...), Δ -> (∇xbagconv(Δ, x, bags,  fs...), nothing,  ∇wbagconv(Δ, x, bags,  fs...)...)
end

"""
	struct BagConv{T, F}
		W::T
		σ::F
	end

	Convolution over a matrix `X` correctly handing borders between bags. The convolution is little bit special, as it 
	assumes that input is always a matrix (never a tensor) and the kernel always spans the full dimension of the vector.

	BagConv(d::Int, o::Int, n::Int, σ = identity)
	`d` --- input dimension
	`o` --- output dimension (number of channels)
	`n` --- size of convolution
	`σ` --- transfer function used after the convolution

	note that of `n` equals one, then the convolution boils down to multiplication of input data `x` with a matrix `W` 
"""
struct BagConv{T, F}
	W::T
	σ::F
end

Flux.@functor BagConv

function BagConv(d::Int, o::Int, n::Int, σ = identity)
	W = (n > 1) ? tuple([randn(o, d) .* sqrt(2.0/(o + d)) for _ in 1:n]...) : randn(o, d) .* sqrt(2.0/(o + d))
	BagConv(W, σ)
end

(m::BagConv{T,F} where {T<:Tuple,F})(x, bags) = m.σ.(bagconv(x, bags, m.W...))
(m::BagConv{T,F} where {T<:AbstractMatrix,F})(x, bags) = m.σ.(m.W * x)
(m::BagConv)(x::ArrayNode, bags::AbstractBags) = ArrayNode(bagconv(x.data, bags, m.W...))
(m::BagConv)(x::BagNode) = ArrayNode(bagconv(x.data.data, x.bags, m.W...))

Base.show(io::IO, m::BagConv) = modelprint(io, m)

function modelprint(io::IO, m::BagConv; pad=[])
  paddedprint(io, "BagConvolution($(size(m.W[1], 2)), $(size(m.W[1], 1)), $(length(m.W)))\n")
end

convsum(bags, xs::AbstractMatrix) = xs
function convsum(bags, xs...)
	offsets = _convshift(length(xs))
	o = similar(xs[1]) .= 0
	for b in bags
		for ri in b 
			for (i, k) in enumerate(offsets)
				if first(b) <= k + ri  <= last(b)
					o[:, ri] .+= view(xs[i], :, k + ri)
				end
			end
		end
	end
	o
end

function ∇convsum(Δ, bags, n)
	offsets = _convshift(n)
	o = [zero(Δ) for i in 1:n]
	for b in bags
		for ri in b 
			for (i, k) in enumerate(offsets)
				if first(b) <= k + ri  <= last(b)
					o[i][:, k + ri] .+= view(Δ, :, ri)
				end
			end
		end
	end
	tuple(o...)
end


Zygote.@adjoint function convsum(bags, xs...)
  convsum(bags, xs...), Δ -> (nothing,  ∇convsum(Δ, bags, length(xs))...)
end

legacy_bagconv(x, bags, f::AbstractArray{T, 3}) where {T} = convsum(bags, [f[:, :, i]*x for i in 1:size(f, 3)]...)
