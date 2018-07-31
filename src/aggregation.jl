function segmented_mean(x::Matrix, bags::Bags)
	o = zeros(eltype(x), size(x, 1), length(bags))
	@inbounds for (j,b) in enumerate(bags)
		for bi in b
			for i in 1:size(x, 1)
				o[i,j] += x[i,bi]
			end
		end
		o[:,j] ./= max(1, length(b))
	end
	o
end


function segmented_mean_back(x::Matrix, bags::Bags, Δ::Matrix)
	o = similar(x, size(x))
	@inbounds for (j,b) in enumerate(bags)
		for bi in b
			for i in 1:size(x,1)
				o[i,bi] = Δ[i,j] / length(b)
			end
		end
	end
	o
end

function segmented_weighted_mean(x::Matrix, bags::Bags, w::Vector)
	o = zeros(eltype(x), size(x, 1), length(bags))
	@inbounds for (j,b) in enumerate(bags)
		ws = sum(w[b])
		for bi in b
			for i in 1:size(x, 1)
				o[i,j] += w[bi] * x[i,bi]
			end
		end
		o[:,j] ./= max(1, ws)
	end
	o
end

function segmented_weighted_mean_back(x::Matrix, bags::Bags, w::Vector, Δ::Matrix)
	o = similar(x, size(x))
	@inbounds for (j, b) in enumerate(bags)
		ws = sum(w[b])
		for bi in b
			for i in 1:size(x,1)
				o[i, bi] = w[bi] * Δ[i, j] / ws
			end
		end
	end
	o
end

function segmented_max(x::Matrix, bags::Bags)
	o = similar(x, size(x,1), length(bags))
	fill!(o, typemin(eltype(x)))
	for (j,b) in enumerate(bags)
		for bi in b
			for i in 1:size(x, 1)
				o[i,j] = max(o[i,j],x[i,bi])
			end
		end
	end
	o[o.== typemin(eltype(x))] .= 0
	o
end

function segmented_max_back(x::Matrix, bags::Bags, Δ::Matrix)
	o = similar(x, size(x))
	fill!(o, 0)
	v = similar(x, size(x,1))
	idxs = zeros(Int,size(x,1))
	@inbounds for (j,b) in enumerate(bags)
		fill!(v, typemin(eltype(x)))
		for bi in b
			for i in 1:size(x, 1)
				if v[i] < x[i,bi]
					idxs[i] = bi
					v[i] = x[i,bi]
				end
			end
		end

		for i in 1:size(x, 1)
			o[i,idxs[i]] = Δ[i,j]
		end
	end
	o
end

function segmented_meanmax(x::Matrix, bags::Bags)
	d = size(x, 1)
	o = similar(x, 2*d, length(bags))
	fill!(view(o, 1:d, :), typemin(eltype(x)))
	fill!(view(o, d+1:2d, :), 0)
	for (j, b) in enumerate(bags)
		for bi in b
			for i in 1:d
				o[i, j] = max(o[i, j], x[i, bi])
				o[i+d, j] += x[i, bi] / max(length(b), 1)
			end
		end
	end
	o
end

function segmented_meanmax_back(x::Matrix, bags::Bags, Δ::Matrix)
	d = size(x, 1)
	o = similar(x, size(x))
	fill!(o, 0)
	v = similar(x, d)
	idxs = zeros(Int, d)
	@inbounds for (j, b) in enumerate(bags)
		fill!(v, typemin(eltype(x)))
		for bi in b
			for i in 1:d
				if v[i] < x[i, bi]
					idxs[i] = bi
					v[i] = x[i, bi]
				end
				o[i, bi] += Δ[i+d, j] / length(b)
			end
		end

		for i in 1:size(x,1)
			o[i,idxs[i]] += Δ[i,j]
		end
	end
	o[ o.== typemin(eltype(x))] .= 0
	o
end


function segmented_weighted_meanmax(x::Matrix, bags::Bags, w::Vector)
	d = size(x, 1)
	o = similar(x, 2*d, length(bags))
	fill!(view(o, 1:d, :), typemin(eltype(x)))
	fill!(view(o, d+1:2d, :), 0)
	for (j, b) in enumerate(bags)
		ws = sum(w[b])
		for bi in b
			for i in 1:d
				o[i, j] = max(o[i, j],x[i, bi])
				o[i+d, j] += w[bi] * x[i, bi] / max(ws, 1)
			end
		end
	end
	o
end

function segmented_weighted_meanmax_back(x::Matrix, bags::Bags, w::Vector, Δ::Matrix)
	d = size(x, 1)
	o = similar(x, size(x))
	fill!(o, 0)
	v = similar(x, d)
	idxs = zeros(Int, d)
	@inbounds for (j, b) in enumerate(bags)
		fill!(v, typemin(eltype(x)))
		ws = sum(w[b])
		for bi in b
			for i in 1:d
				if v[i] < x[i, bi]
					idxs[i] = bi
					v[i] = x[i, bi]
				end
				o[i, bi] += w[bi] * Δ[i+d, j] / ws
			end
		end

		for i in 1:size(x,1)
			o[i, idxs[i]] += Δ[i,j]
		end
	end
	o[o.== typemin(eltype(x))] .= 0
	o
end

segmented_mean(x::ArrayNode, args...) = ArrayNode(segmented_mean(x.data, args...))
segmented_max(x::ArrayNode, args...) = ArrayNode(segmented_max(x.data, args...))
segmented_meanmax(x::ArrayNode, args...) = ArrayNode(segmented_meanmax(x.data, args...))
segmented_weighted_mean(x::ArrayNode, args...) = ArrayNode(segmented_weighted_mean(x.data, args...))
segmented_weighted_max(x::ArrayNode, args...) = ArrayNode(segmented_weighted_max(x.data, args...))
segmented_weighted_meanmax(x::ArrayNode, args...) = ArrayNode(segmented_weighted_max(x.data, args...))

# identical by definition
segmented_weighted_max(x, bags, w) = segmented_max(x, bags)

segmented_mean(x, bags, w) = segmented_mean(x, bags)
segmented_max(x, bags, w) = segmented_max(x, bags)
segmented_meanmax(x, bags, w) = segmented_meanmax(x, bags)

segmented_mean(x::Flux.Tracker.TrackedArray, bags) = Flux.Tracker.track(segmented_mean, x, bags)
Flux.Tracker.back(::typeof(segmented_mean), Δ, x::Flux.Tracker.TrackedArray, bags) = Flux.Tracker.back(x, segmented_mean_back(x.data, bags, Δ))
segmented_weighted_mean(x::Flux.Tracker.TrackedArray, bags, w) = Flux.Tracker.track(segmented_weighted_mean, x, bags, w)
Flux.Tracker.back(::typeof(segmented_weighted_mean), Δ, x::Flux.Tracker.TrackedArray, bags, w) = Flux.Tracker.back(x, segmented_weighted_mean_back(x.data, bags, w, Δ))
segmented_max(x::Flux.Tracker.TrackedArray, bags) = Flux.Tracker.track(segmented_max, x, bags)
Flux.Tracker.back(::typeof(segmented_max), Δ, x::Flux.Tracker.TrackedArray, bags) = Flux.Tracker.back(x, segmented_max_back(x.data, bags, Δ))
segmented_meanmax(x::Flux.Tracker.TrackedArray, bags) = Flux.Tracker.track(segmented_meanmax, x, bags)
Flux.Tracker.back(::typeof(segmented_meanmax), Δ, x::Flux.Tracker.TrackedArray, bags) = Flux.Tracker.back(x, segmented_meanmax_back(x.data, bags, Δ))
segmented_weighted_meanmax(x::Flux.Tracker.TrackedArray, bags, w) = Flux.Tracker.track(segmented_weighted_meanmax, x, bags, w)
Flux.Tracker.back(::typeof(segmented_weighted_meanmax), Δ, x::Flux.Tracker.TrackedArray, bags, w) = Flux.Tracker.back(x, segmented_weighted_meanmax_back(x.data, bags, w, Δ))
