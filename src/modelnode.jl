abstract type MillModel end

"""
	struct ArrayModel <: MillModel
		m::Flux.Chain
	end

	use a Chain on an ArrayNode
"""
struct ArrayModel <: MillModel
	m::Flux.Chain
end

"""
	struct BagModel <: MillModel
		im::MillModel
		a::Function
		bm::MillModel
	end

	use a `im` model on data in `BagNode`, the uses function `a` to aggregate individual bags, 
	and finally it uses `bm` model on the output
"""
struct BagModel <: MillModel
	im::MillModel
	a::Function
	bm::MillModel
end

"""
	struct ProductModel{N} <: MillModel
		ms::NTuple{N, MillModel}
		m::ArrayModel
	end

	uses each model in `ms` on each data in `TreeNode`, concatenate the output and pass it to the chainmodel `m`
"""
struct ProductModel{N} <: MillModel
	ms::NTuple{N, MillModel}
	m::ArrayModel
end

Base.push!(m::ArrayModel, l) = push!(m.m.layers, l)
Base.push!(m::ProductModel, l) = push!(m.m, l)
Base.push!(m::BagModel, l) = push!(m.bm, l)

ArrayModel(f::MillFunction) = ArrayModel(Flux.Chain(f))
BagModel(im::MillFunction, a, bm::MillFunction) = BagModel(ArrayModel(im), a, ArrayModel(bm))
BagModel(im::MillFunction, a) = BagModel(im, a, identity)
BagModel(im::MillModel, a) = BagModel(im, a, ArrayModel(identity))
ProductModel(ms::NTuple{N, MillModel}) where N = ProductModel(ms, ArrayModel(identity))
ProductModel(ms, f::MillFunction) = ProductModel(ms, ArrayModel(f))

Flux.treelike(ArrayModel)
Flux.treelike(BagModel)
Flux.treelike(ProductModel)

Adapt.adapt(T, m::ArrayModel) = ArrayModel(Adapt.adapt(T, m.m))
Adapt.adapt(T, m::BagModel) = BagModel(Adapt.adapt(T, m.im), Adapt.adapt(T, m.a), Adapt.adapt(T, m.bm))
Adapt.adapt(T, m::ProductModel) = ProductModel((map(m -> Adapt.adapt(T, m), m.ms)...,), Adapt.adapt(T, m.m))

(m::ArrayModel)(x::ArrayNode) = ArrayNode(m.m(x.data))
# enforce the same length of ProductModel and TreeNode
(m::BagModel)(x::BagNode) = m.bm(m.a(m.im(x.data), x.bags))
(m::BagModel)(x::WeightedBagNode) = m.bm(m.a(m.im(x.data), x.bags, x.weights))
(m::ProductModel{N})(x::TreeNode{N}) where N = m.m(ArrayNode(vcat(map(f -> f[1](f[2]).data, zip(m.ms, x.data))...)))

################################################################################

function reflectinmodel(x::AbstractBagNode, layerbuilder::Function)
	im, d = reflectinmodel(x.data, layerbuilder)
	bm, d = reflectinmodel(BagModel(im, segmented_meanmax)(x), layerbuilder)
	BagModel(im, segmented_meanmax, bm), d
end

function reflectinmodel(x::AbstractTreeNode, layerbuilder::Function)
	mm = map(i -> reflectinmodel(i, layerbuilder), x.data)
	im = tuple(map(i -> i[1], mm)...)
	# d = mapreduce(i ->i[2], +, mm)
	tm, d = reflectinmodel(ProductModel(im)(x), layerbuilder)
	ProductModel(im, tm), d
end

function reflectinmodel(x::ArrayNode, layerbuilder::Function)
	m = ArrayModel(layerbuilder(size(x.data, 1)))
	m, size(m(x).data, 1)
end

################################################################################

Base.show(io::IO, m::MillModel) = modelprint(io, m)

modelprint(io::IO, m::ArrayModel; offset::Int=0) = paddedprint(io, "$(m.m)\n")

function modelprint(io::IO, m::BagModel; offset::Int=0)
	c = rand(1:256)
	paddedprint(io, "BagModel(\n", color=c)
	paddedprint(io, "↱ ", offset=offset + 2, color=c)
	modelprint(io, m.im, offset=offset + 2)
	paddedprint(io, "↦ ", m.a, "\n", offset=offset + 2, color=c)
	paddedprint(io, "↳ ", offset=offset + 2, color=c)
	modelprint(io, m.bm, offset=offset + 2)
	paddedprint(io, "                )\n", color=c)
end

function modelprint(io::IO, m::ProductModel; offset::Int=0)
	c = rand(1:256)
	paddedprint(io, "ProductModel(\n", color=c)
	for i in 1:length(m.ms)
		paddedprint(io, "⌙ ", offset=offset + 2, color=c)
		modelprint(io, m.ms[1], offset=offset + 4)
	end
	paddedprint(io, "          ) ↦ ", offset=offset + 2, color=c)
	modelprint(io, m.m, offset=offset + 6)
end
