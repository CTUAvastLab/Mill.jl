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



	The main function is `explain(root_data, root_model, ϕ,  explaining_fun)` 
	where 

	`root_data` is a root of a hierarchical explained sample
	`root_model` is a corresponding root of a hierarchical model 
	`ϕ(mask, bagnode, x, mm)` is a function evaluating an effect when instances `x` set to `falses` in `mask` 
			are removed from a `bagnode` and evaluated by a model `mm`. See function `onoff` for an example 
			of such function.

	`explaining_fun(d, f)` estimate importance of `d` components of function `f`. Note that explaining function 
	does not have any clue what `d` and `f` are. They just measure importance


"""
explain(root_data, root_model, ϕ,  explaining_fun) = (d = deepcopy(root_data); explain!(d, root_model, d, root_model, ϕ,  explaining_fun); d)
explain!(root_data, root_model, ϕ,  explaining_fun) = explain!(root_data, root_model, root_data, root_model, ϕ,  explaining_fun)

function explain!(a::BagNode, m::BagModel, root_data, root_model, ϕ,  explaining_fun)
	println("explaining BagNode")
	@show a
	xim = m.im(a.data)
	bagnode = WeightedBagNode(xim, a.bags, fill(true,size(xim.data, 2)))
	x = replacein(root_data, a, bagnode)
	mm = replacein(root_model, m.im, ArrayModel(identity))

	mask = explaining_fun(length(bagnode.weights), mask -> ϕ(mask, bagnode, x, mm))

	println("BagNode: ", ϕ(fill(true,size(xim.data, 2)), bagnode, x, mm)," --> ", ϕ(mask, bagnode, x, mm))
	# @show mask
	sa = removeinstances(a, mask)
	a.data, a.bags, a.metadata = sa.data, sa.bags, sa.metadata
	explain!(a.data, m.im, root_data, root_model, ϕ,  explaining_fun)
	nothing
end

function explain!(a::TreeNode, m::ProductModel, root_data, root_model, ϕ,  explaining_fun)
	for i in 1:length(m.ms)
		explain!(a.data[i], m.ms[i], root_data, root_model, ϕ,  explaining_fun)
	end
	nothing
end

explain!(a, m, root_data, root_model, ϕ,  explaining_fun) = nothing

function onoff(mask, bagnode::WeightedBagNode, x, m, i)
	fill!(bagnode.weights, false); 
	bagnode.weights[mask] .= true; 
	sum(Flux.data(m(x).data)[i,:])
end

explaining_fun(d, f) = explaining_fun(d, f, 0.5, 100)

"""
		function explaining_fun(d, f, p, nprobes)


		d 


"""
function explaining_fun(d, f, p, nprobes)
	daf = Duff.onlinedaf(d, f, p, nprobes)
	scores = Duff.meanscore(daf)
	I = sortperm(scores)
	mask = trues(d)
	println("all features / instances kept ", f(mask))
	for i in 1:d 
		mask[I[i]] = false 
		@show f(mask)
		if f(mask) <=0.5
			mask[I[i]] = true 
			break
		end 
	end
	println("after removing superfluous instances ", f(mask))
	@show mask
	mask
end