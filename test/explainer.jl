using Mill, Flux, Duff

"""
	The explainer works by estimating a Shappley values on the levels of instances. 
	The goal is to identify a "skeleton" of the tree that is responsible for a sample
	having a high score. This, the core of the explainer is explanation of BagNode.

	In Shappley values, we need to have an ability to turn on and off instances. Since the 
	rest of the sample is not interesting, we want to process only a subset.

	We use `replacein` ability to create only and upper part of the model and the datamodel, 
	allows efficient sampling. Also, we replace BagNode with a WeightedBagNode, such that
	we can do masking (WeightedBagNode would need a little more work).

	Once we have prepared upper model with upper data, we can run and estimator of Shappley
	values. We keep only those with positive values

"""
explain(root_data, root_model, ϕ,  explaining_fun) = (d = deepcopy(root_data); explain!(d, root_model, d, root_model, ϕ,  explaining_fun); d)
explain!(root_data, root_model, ϕ,  explaining_fun) = explain!(root_data, root_model, root_data, root_model, ϕ,  explaining_fun)
function explain!(a::BagNode, m::BagModel, root_data, root_model, ϕ,  explaining_fun)
	println("explaining BagNode")
	xim = m.im(a.data)
	bagnode = WeightedBagNode(xim, a.bags, fill(true,size(xim.data, 2)))
	x = replacein(root_data, a, bagnode)
	mm = replacein(root_model, m.im, ArrayModel(identity))

	mask = explaining_fun(length(bagnode.weights), mask -> ϕ(mask, bagnode, x, mm))

	println("BagNode: ", ϕ(fill(true,size(xim.data, 2)), bagnode, x, mm)," --> ", ϕ(mask, bagnode, x, mm))
	explain!(a.data, m.im, root_data, root_model, ϕ,  explaining_fun)
end

function explain!(a::TreeNode, m::ProductModel, root_data, root_model, ϕ,  explaining_fun)
	println("explaining TreeNode")
	for i in 1:length(m.ms)
		a = explain!(root_data, root_model, m,ms[i], a.data[i], ϕ,  explaining_fun)
	end
	a
end

explain!(a, m, root_data, root_model, ϕ,  explaining_fun) = a

function onoff(mask, bagnode::WeightedBagNode, x, m, i)
	fill!(bagnode.weights, false); 
	bagnode.weights[mask] .= true; 
	sum(Flux.data(m(x).data)[i,:])
end

explaining_fun(d, f) = explaining_fun(d, f, 0.5, 100)
function explaining_fun(d, f, p, nprobes)
	daf = Duff.onlinedaf(d, f, p, nprobes)
	scores = Duff.meanscore(daf)
	@show scores
	scores .> 0
end

root_node = BagNode(BagNode(ArrayNode(rand(3,4)),[1:2, 3:4]), [1:2])
root_model = Mill.reflectinmodel(root_node, d -> Dense(d, 4), d -> SegmentedMeanMax())[1]
a, m = root_node.data, root_model.im
ϕ = (x...) -> onoff(x..., 1)

explain!(root_node, root_model, (x...) -> onoff(x..., 1), explaining_fun)

