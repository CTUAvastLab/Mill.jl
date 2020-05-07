struct IdentityModel <: AbstractMillModel
end

(m::IdentityModel)(x) = x

NodeType(::Type{IdentityModel}) = LeafNode()
noderepr(n::IdentityModel) = "IdentityModel"
####
#	We really need partial evaluation !!!
####

function partialeval(m::ArrayModel, ds::ArrayNode, skipnode)
	ds === skipnode && return(m, skipnode)
	return(IdentityModel(), m(ds))
end

# (m::BagModel)(x::WeightedBagNode{<: AbstractNode}) = m.bm(m.a(m.im(x.data), x.bags, x.weights))

function partialeval(m::BagModel, ds::BagNode, skipnode)
	ds === skipnode && return(m, skipnode)
	im, ids = partialeval(m.im, ds.data, skipnode)
	if im === IdentityModel()
		return(IdentityModel(), m.bm(m.a(ids, ds.bags)))
	end
	(BagModel(im, m.a,  m.bm), BagNode(ids, ds.bags, ds.metadata))
end

function partialeval(m::BagModel, ds::WeightedBagNode, skipnode)
	ds === skipnode && return(m, skipnode)
	im, ids = partialeval(m.im, ds.data, skipnode)
	if im === IdentityModel()
		return(IdentityModel(), m.bm(m.a(ids, ds.bags, ds.weights)))
	end
	(BagModel(im, m.a,  m.bm), WeightedBagNode(ids, ds.bags, ds.weights, ds.metadata))
end

function partialeval(m::ProductModel{MS,M}, ds::ProductNode{P,T}, newnode) where {P<:NamedTuple,T,MS<:NamedTuple, M} 
	changed = false 
	ks = keys(m.ms)
	mods = map(ks) do k
		partialeval(m.ms[k], ds.data[k], newnode)
	end
	ms = map(f -> f[1], mods)
	dd = map(f -> f[2], mods)
	if all(f === IdentityModel() for f in ms)
		return(IdentityModel(), m.m(vcat(dd...)))
	end

	return(ProductModel((;zip(ks, ms)...), m.m), ProductNode((;zip(ks, dd)...), ds.metadata))
end

function partialeval(m::ProductModel{MS,M}, ds::ProductNode{P,T}, newnode) where {P<:Tuple,T,MS<:Tuple, M} 
	changed = false 
	mods = map(1:length(m.ms)) do k
		partialeval(m.ms[k], ds.data[k], newnode)
	end
	ms = map(f -> f[1], mods)
	dd = map(f -> f[2], mods)
	if all(f === IdentityModel() for f in ms)
		return(IdentityModel(), m.m(vcat(dd...)))
	end
	return(ProductModel(tuple(ms...), m.m), ProductNode(tuple(dd...), ds.metadata))
end
