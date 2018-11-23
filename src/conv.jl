using Flux
_convshift(n) = (i = div(n, 2); mod(n, 2) == 0 ? (1 - i:i) : (-i : i) )
"""
	_addmatvec!(o, i, W, x, j)

	add an product of matrix `W` with a j-th column of `x` to i-th columns of `o`

"""
function _addmatvec!(o::Matrix, i, W::Matrix, x::Matrix, j)
	for s in 1:size(W, 2)
		for r in 1:size(W, 1)
			o[r, i] += W[r, s] * x[s, j]
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

function bagconv(x, bags, W...)
	offsets = _convshift(length(W))
	o = similar(x, size(W[1], 1), size(x, 2)) .= 0
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
Flux.Tracker.@grad function bagconv(x::Matrix, bags, fs::TrackedMatrix...)
  bagconv(x, bags, Flux.data.(fs)...), Δ -> (nothing, nothing,  ∇wbagconv(Flux.data(Δ), x, bags,  Flux.data.(fs)...)...)
end


struct BagConv{T<:AbstractArray{N,3} where N}
	W::T
end

BagConv(d::Int, o::Int, n::Int) = BagConv(param(randn(o, d, n) .* sqrt(2.0/(o + d))))

(m::BagConv)(x, bags) = bagconv(x, bags, m.W)
(m::BagConv)(x::ArrayNode, bags) = ArrayNode(bagconv(x.data, bags, m.W))


show(io, m::BagConv) = modelprint(io, m)
function modelprint(io::IO, m::BagConv; pad=[])
  c = COLORS[(length(pad)%length(COLORS))+1]
  paddedprint(io, "BagConvolution $(size(m.W))\n", color=c, pad=pad)
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


# """
# 	mutable struct SequentialBagNode{T, C} <: AbstractSequentialBagNode{T, C}
#     data::T
#     bags::Bags
#     metadata::C
# 	end

# 	`SequentialBagNode` is a `BagNode`, where it is assumed that individual instances have some correlation between each other.  
# 	The `SequentialBagNode` is intended to be used with convolution model, followed by the usual pooling `meanmax`. The idea is 
# 	that the convolution will helps to model the correlation.
# """
# mutable struct SequentialBagNode{T, C} <: AbstractBagNode{T, C}
#     data::T
#     bags::Bags
#     metadata::C

#     function SequentialBagNode{T, C}(data::T, bags::Union{Bags, Vector}, metadata::C) where {T <: AbstractNode, C}
#         new(data, bag(bags), metadata)
#     end
# end

# SequentialBagNode(x::T, b::Union{Bags, Vector}, metadata::C=nothing) where {T <: AbstractNode, C} =
# SequentialBagNode{T, C}(x, b, metadata)

# mapdata(f, x::SequentialBagNode) = SequentialBagNode(mapdata(f, x.data), x.bags, x.metadata)

# function _catobs(as::Vector{T}) where {T<:SequentialBagNode}
#   data = _lastcat([x.data for x in as])
#   metadata = _lastcat(filtermetadata([a.metadata for a in as]))
#   bags = _catbags([d.bags for d in as])
#   return SequentialBagNode(data, bags, metadata)
# end

# catobs(as::SequentialBagNode...) = _catobs(collect(as))

# function Base.getindex(x::SequentialBagNode, i::VecOrRange)
#     nb, ii = remapbag(x.bags, i)
#     SequentialBagNode(subset(x.data,ii), nb, subset(x.metadata, i))
# end

# function dsprint(io::IO, n::SequentialBagNode{ArrayNode}; pad=[])
#     c = COLORS[(length(pad)%length(COLORS))+1]
#     paddedprint(io,"SequentialBagNode$(size(n.data)) with $(length(n.bags)) bag(s)\n", color=c)
#     paddedprint(io, "  └── ", color=c, pad=pad)
#     dsprint(io, n.data, pad = [pad; (c, "      ")])
# end

# function dsprint(io::IO, n::SequentialBagNode; pad=[])
#     c = COLORS[(length(pad)%length(COLORS))+1]
#     paddedprint(io,"SequentialBagNode with $(length(n.bags)) bag(s)\n", color=c)
#     paddedprint(io, "  └── ", color=c, pad=pad)
#     dsprint(io, n.data, pad = [pad; (c, "      ")])
# end
