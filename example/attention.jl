# This is an example of how to implement attention in Mill
# Since attention reduce set of vectors to a single vector,
# it is an aggregation and as such, it belongs to aggregation layer
using Mill 
using Flux

""" BagAttention uses `f` to convert the input and `a` to create the attention"""
struct BagAttention{F,A}
	f::F 
	a::A
	agg_f::SegmentedSum
	agg_a::SegmentedSum
end

# remove agg_a from trainable parameters to keep the one in handling missing intact
Flux.trainable(m::BagAttention) = (m.f, m.a, m.agg_f)

function (m::BagAttention)(ds, bags)
	x = m.f(ds.data)
	a = m.a(ds.data)
	am = maximum(a)
	expa = exp.(a .- am)	# to have the stuff safe with respect to overflow
	ArrayNode(m.agg_f(x .* expa, bags) ./ m.agg_a(expa, bags))
end

BagAttention(f, a, agg_f) = BagAttention(f, a, agg_f, SegmentedSum([1f0]))

#Let's create an absolutely dummy dataset 
ds = BagNode(ArrayNode(randn(2,5)), [1:2,2:5,0:-1])

#try it with reflectinmodel
model = reflectinmodel(ds, 
	d -> Dense(d, 4, selu), 
	d -> BagAttention(Dense(d, 4, selu), Dense(d, 1), SegmentedSum(4)),
	)

model(ds)
gradient(() -> sum(sin.(model(ds).data)), Flux.params(model))