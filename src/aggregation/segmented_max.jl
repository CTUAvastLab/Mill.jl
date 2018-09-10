function segmented_max(x::Matrix, bags::Bags)
	o = similar(x, size(x,1), length(bags))
	fill!(o, typemin(eltype(x)))
	for (j,b) in enumerate(bags)
		for bi in b
			for i in 1:size(x, 1)
				o[i, j] = max(o[i, j], x[i, bi])
			end
		end
	end
	o[o.== typemin(eltype(x))] .= 0
	o
end

segmented_max(x::Matrix, bags::Bags, w::Vector) = segmented_max(x, bags)

segmented_max_back(x::TrackedArray, bags::Bags) = Δ -> begin
	x = Flux.data(x)
	o = similar(x, size(x))
	fill!(o, 0.0)
	v = similar(x, size(x,1))
	idxs = zeros(Int, size(x,1))
	@inbounds for (j,b) in enumerate(bags)
		fill!(v, typemin(eltype(x)))
		for bi in b
			for i in 1:size(x, 1)
				if v[i] < x[i, bi]
					idxs[i] = bi
					v[i] = x[i, bi]
				end
			end
		end

		for i in 1:size(x, 1)
			o[i, idxs[i]] = Δ[i, j]
		end
	end
	o, nothing
end

segmented_max_back(x::TrackedArray, bags::Bags, w::Vector) = Δ -> begin
	tuple(segmented_max_back(x, bags)(Δ)..., nothing)
end

segmented_max(x::Flux.Tracker.TrackedArray, args...) = Flux.Tracker.track(segmented_max, x, args...)
segmented_max(x::ArrayNode, args...) = ArrayNode(segmented_max(x.data, args...))

Flux.Tracker.@grad function segmented_max(x, args...)
	segmented_max(Flux.data(x), Flux.data.(args)...), segmented_max_back(x, args...)
end
