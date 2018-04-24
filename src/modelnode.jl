using Flux 

"""
	ModelNode{T}
		m::T
	end

	This model is a counterpart of DataNode. It consists of tuples and apply it on items of DataNode.
"""
struct ModelNode{T}
	m::T
end

Flux.treelike(ModelNode)
Adapt.adapt(T, m::ModelNode) = ModelNode(Adapt.adapt(T,m.m))
(m::ModelNode)(x) = m.m(data(x))
(m::ModelNode{T})(x::V) where {T<:Tuple,V<:Tuple} = vcat(map(f -> f[1](f[2]),zip(m.m,x))...)

"""
	struct AggregationNode{A,B,C}
		im::A 
		a::B 
		bm::C
	end



"""
struct AggregationNode{A,B,C}
	im::A 
	a::B 
	bm::C
end

Flux.treelike(AggregationNode)
Adapt.adapt(T, m::AggregationNode) = AggregationNode(Adapt.adapt(T,m.im),Adapt.adapt(T,m.a),Adapt.adapt(T,m.bm))

(m::AggregationNode)(x::DataNode{A,B,C}) where {A,B<:Bags,C} = m.bm(m.a(m.im(x.data),x.bags))

AggregationNode(im) = AggregationNode(im,identity,identity)
AggregationNode(im,a) = AggregationNode(im,a,identity)

addlayer(m::ModelNode,a) = ModelNode(Flux.Chain(m.m,a))
addlayer(m::AggregationNode{A,B,C},a) where {A,B,C<:Void} = AggregationNode(m.im,m.a,a)
addlayer(m::AggregationNode{A,B,C},a) where {A,B,C} = AggregationNode(m.im,m.a,Flux.Chain(m.bm,a))


function 	reflectinmodel(ds::DataNode{A,B,C},layerbuilder::Function) where {A,B,C}
	(m,d) = reflectinmodel(ds.data,layerbuilder)
	(B<:Bags) ? (AggregationNode(m,segmented_meanmax),2*d) : (ModelNode(m),d)
end

function reflectinmodel(x::A,layerbuilder::Function) where {A<:Tuple} 
	mm = map(i -> reflectinmodel(i,layerbuilder),x)
	im = ModelNode(tuple(map(i -> i[1],mm)...))
	d = mapreduce(i ->i[2],+,mm)
	tm, d = reflectinmodel(im(x),layerbuilder)
	Chain(im,tm),d
end


function reflectinmodel(x::A,layerbuilder::Function) where {A<:AbstractMatrix} 
	m = layerbuilder(size(x,1))
	m, size(m(x),1)
end

