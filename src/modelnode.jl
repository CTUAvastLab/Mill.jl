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

Base.show(io::IO, m::ModelNode{T},offset::Int=0) where {T} = modelprint(io,m,offset)

function modelprint(io::IO, m::ModelNode{T},offset::Int=0) where {T}
	modelprint(io,m.m,offset)
end

function modelprint(io::IO, m::Flux.Chain,offset::Int=0)
	paddedprint(io,"",offset)
	for i in 1:length(m.layers)
		modelprint(io,m.layers[i],offset);
	end
end

function modelprint(io::IO, m::ModelNode{T},offset::Int=0) where {T<:Tuple}
	print(io,"\nTuple:\n")
	for i in 1:length(m.m)
		modelprint(io,m.m[i],offset+2);
	end
end

function modelprint(io,m,offset)
	paddedprint(io,"",offset) 
	print(io,m)
	println()
end

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

function Base.show(io::IO, m::AggregationNode)
	print(io,"AggregationNode(")
	print(io,m.im)
	print(io," meanmax")
	print(io,m.bm)
	print(io,")")
end
Base.show(io::IO, m::AggregationNode,offset::Int=0) = modelprint(io,m,offset)

function modelprint(io::IO, m::AggregationNode,offset::Int=0)
	paddedprint(io,"AggregationNode\n",offset)
	modelprint(io,m.im,offset+2)
	paddedprint(io,"meanmax",offset+2)
	print(io,"\n")
	if m.bm != identity
		modelprint(io,m.bm,offset+2)
	end
end

function 	reflectinmodel(ds::DataNode{A,B,C},layerbuilder::Function) where {A,B,C}
	(m,d) = reflectinmodel(ds.data,layerbuilder)
	if B<:Bags
		mb,d =  reflectinmodel(segmented_meanmax(m(ds.data),ds.bags),layerbuilder)
		return(AggregationNode(m,segmented_meanmax,mb),d)
	else 
		return (ModelNode(m),d)
	end
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

