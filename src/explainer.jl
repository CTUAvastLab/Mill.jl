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



	The main function is `explain(root_data, root_model, ϕ,  explaining_fun, recurse_depth)` 
	where 

	`root_data` is a root of a hierarchical explained sample
	`root_model` is a corresponding root of a hierarchical model 
	`ϕ(mask, bagnode, x, mm)` is a function evaluating an effect when instances `x` set to `falses` in `mask` 
			are removed from a `bagnode` and evaluated by a model `mm`. See function `onoff` for an example 
			of such function.

	`explaining_fun(d, f)` estimate importance of `d` components of function `f`. Note that explaining function 
	does not have any clue what `d` and `f` are. They just measure importance


"""
function explain(ds, model, i)
	sub_ds = deepcopy(ds)
	for index in setdiff(list(model),"")
		try
			sub_ds = explain(sub_ds[index], model[index], sub_ds, model, (x...) -> onoff(x..., i), explaining_fun)
		catch
		end
	end
	sub_ds
end

function explain(a::BagNode, m::BagModel, root_data, root_model, ϕ,  explaining_fun)
	# println("explaining BagNode")
	nobs(a) == 0 && return(root_data)
	xim = m.im(a.data)
	bagnode = WeightedBagNode(xim, a.bags, fill(true,size(xim.data, 2)))
	x = replacein(root_data, a, bagnode)
	mm = replacein(root_model, m.im, ArrayModel(identity))

	mask = explaining_fun(length(bagnode.weights), mask -> ϕ(mask, bagnode, x, mm))
	sa = removeinstances(a, mask)
	# @show sa
	replacein(root_data, a, sa)
end

function explain(a::TreeNode, m::ProductModel, root_data, root_model, ϕ,  explaining_fun)
	root_data
end

function explain(a::ArrayNode{T,A}, m::ArrayModel, root_data, root_model, ϕ,  explaining_fun) where {T<:SparseMatrixCSC, A}
	x = a.data
	nnz(x) == 0 && return(root_data)
	original_x = deepcopy(x.nzval)

	mask = explaining_fun(nnz(x), 
		mask -> ϕ(mask, x.nzval, original_x, root_data, root_model)
		)
	x.nzval[ .! mask] .= 0
	root_data
end

function explain(a::ArrayNode{T,A}, m::ArrayModel, root_data, root_model, ϕ,  explaining_fun) where {T<:Matrix, A}
	x = a.data
	all(x .== 0) && return(root_data)
	original_x = deepcopy(x)

	mask = explaining_fun(nnz(x), 
		mask -> ϕ(mask, x.nzval, original_x, root_data, root_model)
		)
	x.nzval[ .! mask] .= 0
	root_data
end

function onoff(mask, bagnode::WeightedBagNode, ds, model, i)
	fill!(bagnode.weights, false); 
	bagnode.weights[mask] .= true; 
	sum(model(ds).data[i,:])
end

function onoff(mask, xe::AbstractArray, original_x::AbstractArray, ds, model, i)
	xe .= 0
	xe[mask] .= original_x[mask]
	o = sum(model(ds).data[i,:])
	xe .= original_x
	o
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
	# @show scores
	mask = trues(d)
	if f(mask) <= 0.5 
		# @info "nothing to explain"
		return(mask)
	end
	# @info "before removing feature / intances $(f(mask))" 
	for i in 1:d 
		mask[I[i]] = false 
		if f(mask) <=0.5
			mask[I[i]] = true 
			break
		end 
	end
	# @show mask
	# @info "after removing feature / intances $(f(mask))"
	isnan(f(mask)) && @error "f(mask) is nan"
	mask
end