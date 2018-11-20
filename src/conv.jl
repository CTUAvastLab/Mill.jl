using Flux
_convshift(n) = (i = div(n, 2); mod(n, 2) == 0 ? (1 - i:i) : (-i : i) )

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

convmil(x, bags, f::AbstractArray{T, 3}) where {T} = convsum(bags, [f[:, :, i]*x for i in 1:size(f, 3)]...)

convsum(bags, xs::TrackedMatrix...) = Flux.Tracker.track(convsum, bags, xs...)
Flux.Tracker.@grad function convsum(bags, xs::TrackedMatrix...)
  convsum(bags, Flux.data.(xs)...), Δ -> (nothing,  ∇convsum(Flux.data(Δ), bags, length(xs))...)
end

