# This is an example of how to hack a self attention in Mill
# Since attention reduce set of vectors to a single vector,
# it is an aggregation and as such, it belongs to aggregation layer
using Mill 
using Flux
using Transformers
using Transformers.Basic: MultiheadAttention

""" BagAttention uses `f` to convert the input and `a` to create the attention"""
struct BagMultiheadSelfAttention{F,A}
	mh::F 
	agg::A
end

# remove agg_a from trainable parameters to keep the one in handling missing intact
Flux.@functor BagMultiheadSelfAttention

function (m::BagMultiheadSelfAttention)(ds, bags)
	xs = map(b -> m.mh(ds.data[:,b],ds.data[:,b],ds.data[:,b]), filter(!isempty, bags.bags))
	ArrayNode(m.agg(reduce(hcat, xs), bags))
end

function BagMultiheadSelfAttention(input_dim::Int, nheads::Int, hidden_dim::Int, odim::Int)
		BagMultiheadSelfAttention(
			MultiheadAttention(nheads, input_dim, hidden_dim, odim; future=false, pdrop = 0),
			SegmentedMeanMax(odim))
end

#Let's create an absolutely dummy dataset 
ds = BagNode(randn(2,5), [1:2,2:5,0:-1])

#try it with reflectinmodel
model = reflectinmodel(ds, 
	d -> Dense(d, 4, selu), 
	d -> BagMultiheadSelfAttention(d, 3, 3, 5),
	)

model(ds)
gradient(() -> sum(sin.(model(ds).data)), Flux.params(model))
