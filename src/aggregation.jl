
function segmented_mean(x,bags)
	o = similar(x,size(x,1),length(bags))
	fill!(o,0)
	@inbounds for (j,b) in enumerate(bags)
		for bi in b
			for i in 1:size(x,1)
				o[i,j] += x[i,bi]
			end 
		end
		o[:,j] ./= length(b)
	end
	o
end

function segmented_mean_back(x,bags,Δ)
	o = similar(x,size(x))
	@inbounds for (j,b) in enumerate(bags)
		for bi in b
			for i in 1:size(x,1)
				o[i,bi] = Δ[i,j] / length(b)
			end 
		end
	end
	o
end

function segmented_max(x,bags)
	o = similar(x,size(x,1),length(bags))
	fill!(o,typemin(eltype(x)))
	for (j,b) in enumerate(bags)
		for bi in b
			for i in 1:size(x,1)
				o[i,j] = max(o[i,j],x[i,bi])
			end 
		end
	end
	o
end

function segmented_max_back(x,bags,Δ)
	o = similar(x,size(x))
	fill!(o,0)
	v = similar(x,size(x,1))
	idxs = zeros(Int,size(x,1))
	@inbounds for (j,b) in enumerate(bags)
		fill!(v,typemin(eltype(x)))
		for bi in b
			for i in 1:size(x,1)
				if v[i] < x[i,bi] 
					idxs[i] = bi
					v[i] = x[i,bi]
				end
			end 
		end

		for i in 1:size(x,1)
			o[i,idxs[i]] = Δ[i,j] 
		end
	end
	o
end


function segmented_meanmax(x,bags)
	d = size(x,1)
	o = similar(x,2*d,length(bags))
	fill!(view(o,1:d,:),typemin(eltype(x)))
	fill!(view(o,d+1:2d,:),0)
	for (j,b) in enumerate(bags)
		for bi in b
			for i in 1:d
				o[i,j] = max(o[i,j],x[i,bi])
				o[i+d,j] += x[i,bi]/length(b)
			end 
		end
	end
	o
end

function segmented_meanmax_back(x,bags,Δ)
	d = size(x,1)
	o = similar(x,size(x))
	fill!(o,0)
	v = similar(x,d)
	idxs = zeros(Int,d)
	@inbounds for (j,b) in enumerate(bags)
		fill!(v,typemin(eltype(x)))
		for bi in b
			for i in 1:d
				if v[i] < x[i,bi] 
					idxs[i] = bi
					v[i] = x[i,bi]
				end
				o[i,bi] += Δ[i+d,j] / length(b)
			end 
		end

		for i in 1:size(x,1)
			o[i,idxs[i]] += Δ[i,j] 
		end
	end
	o
end


segmented_mean(x::Flux.Tracker.TrackedArray, bags) = Flux.Tracker.track(segmented_mean, x, bags)
Flux.Tracker.back(::typeof(segmented_mean), Δ, x::Flux.Tracker.TrackedArray, bags) = Flux.Tracker.back(x, segmented_mean_back(x.data,bags,Δ))
segmented_max(x::Flux.Tracker.TrackedArray, bags) = Flux.Tracker.track(segmented_max, x, bags)
Flux.Tracker.back(::typeof(segmented_max), Δ, x::Flux.Tracker.TrackedArray, bags) = Flux.Tracker.back(x, segmented_max_back(x.data,bags,Δ))
segmented_meanmax(x::Flux.Tracker.TrackedArray, bags) = Flux.Tracker.track(segmented_meanmax, x, bags)
Flux.Tracker.back(::typeof(segmented_meanmax), Δ, x::Flux.Tracker.TrackedArray, bags) = Flux.Tracker.back(x, segmented_meanmax_back(x.data,bags,Δ))
