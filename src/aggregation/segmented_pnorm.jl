# https://arxiv.org/pdf/1311.1780.pdf
struct PNorm{T}
	ρ::T
	c::T
end

PNorm(d::Integer) = PNorm(Flux.Tracker.param(randn(d)), Flux.Tracker.param(randn(d)))
Flux.treelike(PNorm)

(n::PNorm)(x::Matrix, bags) = segmented_pnorm(x, bags, p_map(Flux.data(n.ρ)), Flux.data(n.c))
(n::PNorm)(x::TrackedArray, bags) = segmented_pnorm(x, bags, p_map(n.ρ), n.c)

p_map(ρ) = 1 .+ log.(1 .+ exp.(ρ))
inv_p_map(p) = log.(exp.(p-1) .- 1)

function segmented_pnorm(x::Matrix, bags::Bags, p::Vector, c::Vector)
	@show typeof(x)
	@show typeof(p)
	@show typeof(c)
	o = zeros(eltype(p), size(x, 1), length(bags))
	@inbounds for (j, b) in enumerate(bags)
		for bi in b
			for i in 1:size(x, 1)
				o[i, j] += abs(x[i, bi] - c[i]) ^ p[i]
			end
		end
		o[:, j] ./= max(1, length(b))
		o[:, j] .^= 1 ./ p
	end
	o
end

segmented_pnorm_back(x::TrackedArray, n::Matrix, bags::Bags, p::Vector, c::Vector) = Δ -> begin
@show typeof(x)
@show typeof(p)
@show typeof(c)
	dx = zeros(eltype(p), size(x))
	@inbounds for (j, b) in enumerate(bags)
		for bi in b
			for i in 1:size(x,1)
				dx[i, bi] = Δ[i, j] * sign(x[i, bi] - c[i])
				dx[i, bi] /= max(1, length(b))
				dx[i, bi] *= (abs(x[i, bi] - c[i]) / n[i, j]) ^ (p[i] - 1)
			end
		end
	end
	dx, nothing, nothing , nothing
end

segmented_pnorm_back(x::TrackedArray, n::Matrix, bags::Bags, p::TrackedVector, c::TrackedVector) = Δ -> begin
	println("segmented_pnorm_back")
	@show typeof(x)
	@show typeof(p)
	@show typeof(c)
	dx = zeros(eltype(p), size(x))
	dp = zeros(eltype(p), size(p))
	dps1 = zeros(eltype(p), size(p))
	dps2 = zeros(eltype(p), size(p))
	dc = zeros(eltype(c), size(c))
	dcs = zeros(eltype(c), size(c))
	@inbounds for (j, b) in enumerate(bags)
		dcs .= 0
		dps1 .= 0
		dps2 .= 0
		for bi in b
			for i in 1:size(x,1)
				ab = abs(x[i, bi] - c[i])
				sig = sign(x[i, bi] - c[i])
				dx[i, bi] = Δ[i, j] * sig
				dx[i, bi] /= max(1, length(b))
				dx[i, bi] *= (ab / n[i, j]) ^ (p[i] - 1)
				dps1[i] +=  ab ^ p[i] * log(ab)
				dps2[i] +=  ab ^ p[i]
				dcs[i] -= sig * (ab ^ (p[i] - 1))
			end
		end
		t = n[:, j] ./ p .* (dps1 ./ dps2 .- (log.(dps2) .- log(max(1, length(b)))) ./ p)
		dp .+= Δ[:, j] .* t
		dcs ./= max(1, length(b))
		dcs .*= n[:, j] .^ (1 .- p)
		dc .+= Δ[:, j] .* dcs
	end
	dx, nothing, dp , dc
end

function segmented_pnorm_back(x::TrackedArray, n::Matrix, bags::Bags, p::TrackedVector, c::Vector)
	error("Not supported yet")
end

function segmented_pnorm_back(x::TrackedArray, n::Matrix, bags::Bags, p::Vector, c::TrackedVector)
	error("Not supported yet")
end

Flux.Tracker.@grad function segmented_pnorm(x, args...)
	n = segmented_pnorm(Flux.data(x), Flux.data.(args)...)
	grad = segmented_pnorm_back(x, n, args...)
	n, grad
end
