abstract type MillModel end

struct ChainModel <: MillModel
	m::Flux.Chain
end

struct JointModel{N} <: MillModel
	m::Flux.Chain
	ms::NTuple{N, MillModel}
end

struct AggregationModel <: MillModel
	im::MillModel
	a::Function
	bm::MillModel
end

AggregationModel(im) = AggregationModel(im, identity, identity)
AggregationModel(im, a) = AggregationModel(im, a, identity)
JointModel(ms::Tuple{MillModel}) = JointModel(identity, ms)

Flux.treelike(ChainModel)
Flux.treelike(JointModel)
Flux.treelike(AggregationModel)

Adapt.adapt(T, m::ChainModel) = ChainModel(Adapt.adapt(T, m.m))
Adapt.adapt(T, m::JointModel) = JointModel(Adapt.adapt(T, m.m), (map(m -> Adapt.adapt(T, m), m.ms)...))
Adapt.adapt(T, m::AggregationModel) = AggregationModel(Adapt.adapt(T, m.im), Adapt.adapt(T, m.a), Adapt.adapt(T, m.bm))

(m::ChainModel)(x::MatrixNode) = m.m(x.data)
# enforce the same length of JointModel and TreeNode
(m::JointModel{N})(x::TreeNode{N}) where N = m.m(vcat(map(f -> f[1](f[2]), zip(m.ms, x.data))...))
(m::AggregationModel)(x::BagNode) = m.bm(m.a(m.im(x.data), x.bags))
(m::AggregationModel)(x::WeightedBagNode) = m.bm(m.a(m.im(x.data), x.bags, x.weights))

################################################################################

function reflectinmodel(x::AbstractBag, layerbuilder::Function)
	im, d = reflectinmodel(x.data, layerbuilder)
	bm, d =  reflectinmodel(Matrix(segmented_meanmax(im(x.data), x.bags)), layerbuilder)
	Aggregation(im, segmented_meanmax, bm), d
end

function reflectinmodel(x::AbstractTree, layerbuilder::Function)
	mm = map(i -> reflectinmodel(i, layerbuilder), x.data)
	im = tuple(map(i -> i[1], mm)...)
	# d = mapreduce(i ->i[2], +, mm)
	tm, d = reflectinmodel(Matrix(Model(im, identity)(x)), layerbuilder)
	Model(im, tm), d
end

reflectinmodel(x::Matrix, layerbuilder::Function) = reflectinmodel(x.data, layerbuilder)

function reflectinmodel(x::AbstractMatrix, layerbuilder::Function)
	m = layerbuilder(size(x, 1))
	Model(m, identity), size(m(x), 1)
end

################################################################################

Base.show(io::IO, m::MillModel, offset::Int=0) = modelprint(io, m, offset)

function modelprint(io::IO, m::ChainModel, offset::Int=0)
	paddedprint()
	modelprint(io, m.m, offset)
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

function modelprint(io::IO,  m::AggregationNode, offset::Int=0)
	paddedprint(io, "AggregationNode\n", offset)
	modelprint(io, m.im, offset+2)
	paddedprint(io, "meanmax", offset+2)
	print(io, "\n")
	m.bm != identity &&	modelprint(io, m.bm, offset+2)
end
