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

function segmented_mean(x::Matrix, bags::Bags, w::Vector)
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

segmented_mean_back(x::TrackedArray, bags::Bags) = Δ -> begin
	dx = similar(x, size(x))
	@inbounds for (j,b) in enumerate(bags)
		for bi in b
			for i in 1:size(x,1)
				dx[i,bi] = Δ[i,j] / length(b)
			end
		end
	end
	dx, nothing
end


segmented_mean_back(x::TrackedArray, bags::Bags, w::Vector) = Δ -> begin
	dx = similar(x, size(x))
	@inbounds for (j, b) in enumerate(bags)
		ws = sum(w[b])
		for bi in b
			for i in 1:size(x,1)
				dx[i, bi] = w[bi] * Δ[i, j] / ws
			end
		end
	end
	dx, nothing, nothing
end

segmented_mean_back(x::TrackedArray, bags::Bags, w::TrackedVector) = Δ -> begin
	error("Not supported yet")
end

segmented_mean_back(x::Matrix, bags::Bags, w::TrackedVector) = Δ -> begin
	error("Not supported yet")
end
