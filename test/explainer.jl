using Mill, Flux, Duff

"""
	The explainer works by estimating a Shappley values on the levels of instances. 
	The goal is to identify a "skeleton" of the tree that is responsible for a sample
	having a high score. This, the core of the explainer is explanation of BagNode.

	In Shappley values, we need to have an ability to turn on and off instances. Since the 
	rest of the sample is not interesting, we want to process only a subset.

	We use `replacein` ability to create only and upper part of the model.
	But we need to have a similar functionality for datasets.

"""
function explain(root_model, root_data, m::BagNode, a)
	xim = m.im(a.data)
	bagnode = WeightedBagNode(xim, a.bags, fill(true,size(xim.data, 2)))
	x = replacein(root_data, a, bagnode)
	mm = replacein(root_model, m.im, ArrayModel(identity))
	daf = onlinedaf(length(bagnode.weights), mask -> onoff(mask, bagnode, mm), p, nprobes)
end

function onoff(mask, b::WeightedBagNode, m)
	fill!(b.weights, true); 
	b.weights[mask] .= false; 
	Flux.data(m(x).data[1])
end

a = BagNode(BagNode(ArrayNode(rand(3,4)),[1:2, 3:4]), [1:2])
m = Mill.reflectinmodel(a, d -> Dense(d, 4), d -> SegmentedMeanMax())[1]

