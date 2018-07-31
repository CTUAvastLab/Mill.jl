abstract type MillModel end

struct ChainModel <: MillModel
	m::Flux.Chain
end

struct AggregationModel <: MillModel
	im::MillModel
	a::Function
	bm::MillModel
end

struct JointModel{N} <: MillModel
	ms::NTuple{N, MillModel}
	m::ChainModel
end

ChainModel(f::MillFunction) = ChainModel(Flux.Chain(f))
AggregationModel(im::MillFunction, a, bm::MillFunction) = AggregationModel(ChainModel(im), a, ChainModel(bm))
AggregationModel(im::MillFunction, a) = AggregationModel(im, a, identity)
AggregationModel(im::MillModel, a) = AggregationModel(im, a, ChainModel(identity))
JointModel(ms::NTuple{N, MillModel}) where N = JointModel(ms, ChainModel(identity))
JointModel(ms, f::MillFunction) = JointModel(ms, ChainModel(f))

Flux.treelike(ChainModel)
Flux.treelike(AggregationModel)
Flux.treelike(JointModel)

Adapt.adapt(T, m::ChainModel) = ChainModel(Adapt.adapt(T, m.m))
Adapt.adapt(T, m::AggregationModel) = AggregationModel(Adapt.adapt(T, m.im), Adapt.adapt(T, m.a), Adapt.adapt(T, m.bm))
Adapt.adapt(T, m::JointModel) = JointModel((map(m -> Adapt.adapt(T, m), m.ms)...), Adapt.adapt(T, m.m))

(m::ChainModel)(x::ArrayNode) = ArrayNode(m.m(x.data))
# enforce the same length of JointModel and TreeNode
(m::AggregationModel)(x::BagNode) = m.bm(m.a(m.im(x.data), x.bags))
(m::AggregationModel)(x::WeightedBagNode) = m.bm(m.a(m.im(x.data), x.bags, x.weights))
(m::JointModel{N})(x::TreeNode{N}) where N = m.m(ArrayNode(vcat(map(f -> f[1](f[2]).data, zip(m.ms, x.data))...)))

################################################################################

function reflectinmodel(x::AbstractBagNode, layerbuilder::Function)
	im, d = reflectinmodel(x.data, layerbuilder)
	bm, d = reflectinmodel(AggregationModel(im, segmented_meanmax)(x), layerbuilder)
	AggregationModel(im, segmented_meanmax, bm), d
end

function reflectinmodel(x::AbstractTreeNode, layerbuilder::Function)
	mm = map(i -> reflectinmodel(i, layerbuilder), x.data)
	im = tuple(map(i -> i[1], mm)...)
	# d = mapreduce(i ->i[2], +, mm)
	tm, d = reflectinmodel(JointModel(im)(x), layerbuilder)
	JointModel(im, tm), d
end

function reflectinmodel(x::ArrayNode, layerbuilder::Function)
	m = ChainModel(layerbuilder(size(x.data, 1)))
	m, size(m(x).data, 1)
end

################################################################################

Base.show(io::IO, m::MillModel) = modelprint(io, m)

modelprint(io::IO, m::ChainModel; offset::Int=0) = paddedprint(io, "$(m.m)\n")

function modelprint(io::IO, m::AggregationModel; offset::Int=0)
	c = rand(1:256)
	paddedprint(io, "AggregationModel(\n", color=c)
	paddedprint(io, "↱ ", offset=offset + 2, color=c)
	modelprint(io, m.im, offset=offset + 2)
	paddedprint(io, "↦ ", m.a, "\n", offset=offset + 2, color=c)
	paddedprint(io, "↳ ", offset=offset + 2, color=c)
	modelprint(io, m.bm, offset=offset + 2)
	paddedprint(io, "                )\n", color=c)
end

function modelprint(io::IO, m::JointModel; offset::Int=0)
	c = rand(1:256)
	paddedprint(io, "JointModel(\n", color=c)
	for i in 1:length(m.ms)
		paddedprint(io, "⌙ ", offset=offset + 2, color=c)
		modelprint(io, m.ms[1], offset=offset + 4)
	end
	paddedprint(io, "          ) ↦ ", offset=offset + 2, color=c)
	modelprint(io, m.m, offset=offset + 6)
end
