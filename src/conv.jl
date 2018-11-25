using Flux
_convshift(n) = (i = div(n, 2); mod(n, 2) == 0 ? (1 - i:i) : (-i : i) )
"""
	_addmatvec!(o, i, W, x, j)

	add an product of matrix `W` with a j-th column of `x` to i-th columns of `o`

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
# _addmatvec!(o, i, W, x, j) = o[:, i] .+=  W * view(x, :, j)


"""
	_outeradd!(W, Δ, i, x, j)

	add an outer product of i-th column of `Δ` and transposed `j`-th columns of `x` to `W`

"""
function _addvecvect!(W::Matrix, Δ::Matrix, i, x::Matrix, j)
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
	for (r, v) in zip(rowptr, vals)
		for s in 1:size(W, 1)
			W[s, r] += Δ[s, i] * v
		end
	end
end

function bagconv(x, bags, W...)
	offsets = _convshift(length(W))
	o = similar(W[1], size(W[1], 1), size(x, 2)) .= 0
	for b in bags
		for ri in b 
			for (i, k) in enumerate(offsets)
				if first(b) <= k + ri  <= last(b)
					_addmatvec!(o, ri, W[i], x, k + ri)
				end
			end
		end
	end
	o
end

function ∇wbagconv(Δ, x, bags, W...)
	offsets = _convshift(length(W))
	∇W = [similar(w) .= 0 for w in W]
	for b in bags
		for ri in b 
			for (i, k) in enumerate(offsets)
				if first(b) <= k + ri  <= last(b)
					_addvecvect!(∇W[i], Δ, ri, x, k + ri)
				end
			end
		end
	end
	tuple(∇W...)
end

bagconv(x, bags, fs::TrackedMatrix...) = Flux.Tracker.track(bagconv, x, bags, fs...)
Flux.Tracker.@grad function bagconv(x, bags, fs::TrackedMatrix...)
  bagconv(x, bags, Flux.data.(fs)...), Δ -> (nothing, nothing,  ∇wbagconv(Flux.data(Δ), x, bags,  Flux.data.(fs)...)...)
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

Flux.@treelike BagConv

function BagConv(d::Int, o::Int, n::Int, σ = identity)
	W = (n > 1) ? tuple([param(randn(o, d) .* sqrt(2.0/(o + d))) for _ in 1:n]...) : param(randn(o, d) .* sqrt(2.0/(o + d)))
	BagConv(W, σ)
end

(m::BagConv{T,F} where {T<:Tuple,F})(x, bags) = m.σ.(bagconv(x, bags, m.W...))
(m::BagConv{T,F} where {T<:AbstractMatrix,F})(x, bags) = m.σ.(m.W * x)
(m::BagConv)(x::ArrayNode, bags) = ArrayNode(bagconv(x.data, bags, m.W...))


show(io, m::BagConv) = modelprint(io, m)
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
	o = [similar(Δ) .= 0 for i in 1:n]
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


convsum(bags, xs::TrackedMatrix...) = Flux.Tracker.track(convsum, bags, xs...)
Flux.Tracker.@grad function convsum(bags, xs::TrackedMatrix...)
  convsum(bags, Flux.data.(xs)...), Δ -> (nothing,  ∇convsum(Flux.data(Δ), bags, length(xs))...)
end

legacy_bagconv(x, bags, f::AbstractArray{T, 3}) where {T} = convsum(bags, [f[:, :, i]*x for i in 1:size(f, 3)]...)