using Test
using Flux.Tracker: TrackedReal, gradcheck, grad, checkpoint
gradtest(f, xs::AbstractArray...) = gradcheck((xs...) -> sum(sin.(f(xs...))), xs...)

_convshift(n) = (i = div(n, 2); mod(n, 2) == 0 ? (1 - i:i) : (-i : i) )


convsum(bags, xs) = xs
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

function convolution(x, bags, f::AbstractArray{3})
	@assert size(x, 1) == size(f, 1)
	convsum(bags, [f[:, :, i] for i in 1:size(f, 3)]...)
end


@testset "testing convolution shift" begin
	@test _convshift(2) == 0:1
	@test _convshift(3) == -1:1
	@test _convshift(4) == -1:2
	@test _convshift(5) == -2:2
end

@testset "testing forward convolution & gradient" begin
	x = [1 10  100  1000  10000];
	y = 2 .* x;
	z = 4 .* x;
	bags = [1:2,3:5];

	@test convsum(bags, x) == x
	@test convsum(bags, x, y) == [21  10  2100  21000  10000]
	@test convsum(bags, x, y, z) == [42  21  4200  42100  21000]

	all(Flux.Tracker.ngradient(x -> sum(convsum(bags,x, y, z)), x)[1] .== ∇convsum(Δ, bags, 3)[1])
	all(Flux.Tracker.ngradient(y -> sum(convsum(bags,x, y, z)), y)[1] .== ∇convsum(Δ, bags, 3)[2])
	all(Flux.Tracker.ngradient(z -> sum(convsum(bags,x, y, z)), z)[1] .== ∇convsum(Δ, bags, 3)[3])
end