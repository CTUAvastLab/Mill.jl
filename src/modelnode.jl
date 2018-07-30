abstract type ModelNode end

struct ChainNode <: ModelNode
	f::Flux.Chain
end

struct CatNode <: ModelNode
end

struct AggregationNode <: ModelNode
	im::ChainNode
	a::Function
	bm::ChainNode
end

Flux.treelike(ModelNode)
Adapt.adapt(T, m::ModelNode) = ModelNode(Adapt.adapt(T, m.f), Adapt.adapt(T, m.m))

(m::ModelNode)(x::MatrixNode) = m(x.data)
(m::ModelNode{F, T})(x::TreeNode) where {F<:Tuple, T} = m.m(vcat(map(f -> f[1](f[2]), zip(m.f, x.data))...))
(m::ModelNode)(x::AbstractMatrix) = m.m(m.f(x))

addlayer(m::ModelNode{A, B}, a) where {A, B<:Chain} = ModelNode(m.f, Flux.Chain(vcat(m.m.layers, [a])...))
addlayer(m::ModelNode, a) = (m.m == identity)? ModelNode(m.f, a) : ModelNode(m.f, Flux.Chain([m.m, a]...))

Base.show(io::IO, m::ModelNode, offset::Int=0) = modelprint(io, m, offset)

function modelprint(io::IO, m::ModelNode, offset::Int=0)
	modelprint(io, m.f, offset)
	m.m != identity && modelprint(io, m.m, offset)
end

function modelprint(io::IO, m::Flux.Chain, offset::Int=0)
	paddedprint(io, "", offset)
	for i in 1:length(m.layers)
		modelprint(io, m.layers[i], offset);
	end
end

function modelprint(io::IO, m::ModelNode{<:Tuple, M}, offset::Int=0) where M
	paddedprint(io, "Tuple:\n", offset)
	for i in 1:length(m.f)
		modelprint(io, m.f[i], offset+2);
	end
	m.m != identity && modelprint(io, m.m, offset);
end

function modelprint(io, m, offset)
	paddedprint(io, "", offset)
	print(io, m, "\n")
end

Flux.treelike(AggregationNode)
Adapt.adapt(T,  m::AggregationNode) = AggregationNode(Adapt.adapt(T, m.im), Adapt.adapt(T, m.a), Adapt.adapt(T, m.bm))

(m::AggregationNode)(x::BagNode) = m.bm(m.a(m.im(x.data), x.bags))
(m::AggregationNode)(x::WeightedBagNode) = m.bm(m.a(m.im(x.data), x.bags, x.weights))

AggregationNode(im) = AggregationNode(im, identity, identity)
AggregationNode(im, a) = AggregationNode(im, a, identity)

addlayer(m::AggregationNode{A, B, C}, a) where {A, B, C<:Void} = AggregationNode(m.im, m.a, a)
addlayer(m::AggregationNode, a) = AggregationNode(m.im, m.a, Flux.Chain(m.bm, a))

Base.show(io::IO,  m::AggregationNode, offset::Int=0) = modelprint(io, m, offset)

function modelprint(io::IO,  m::AggregationNode, offset::Int=0)
	paddedprint(io, "AggregationNode\n", offset)
	modelprint(io, m.im, offset+2)
	paddedprint(io, "meanmax", offset+2)
	print(io, "\n")
	m.bm != identity &&	modelprint(io, m.bm, offset+2)
end

function reflectinmodel(x::AbstractBagNode, layerbuilder::Function)
	im, d = reflectinmodel(x.data, layerbuilder)
	bm, d =  reflectinmodel(MatrixNode(segmented_meanmax(im(x.data), x.bags)), layerbuilder)
	AggregationNode(im, segmented_meanmax, bm), d
end

function reflectinmodel(x::AbstractTreeNode, layerbuilder::Function)
	mm = map(i -> reflectinmodel(i, layerbuilder), x.data)
	im = tuple(map(i -> i[1], mm)...)
	# d = mapreduce(i ->i[2], +, mm)
	tm, d = reflectinmodel(MatrixNode(ModelNode(im, identity)(x)), layerbuilder)
	ModelNode(im, tm), d
end

reflectinmodel(x::MatrixNode, layerbuilder::Function) = reflectinmodel(x.data, layerbuilder)

function reflectinmodel(x::AbstractMatrix, layerbuilder::Function)
	m = layerbuilder(size(x, 1))
	ModelNode(m, identity), size(m(x), 1)
end
