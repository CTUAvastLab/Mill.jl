using LightGraphs

abstract type AbstractGraphModel <: AbstractMillModel end

struct GraphModel{V,E,M,R} <: AbstractGraphModel
	vlifter::V
	elifter::E
	messager::M
	reducer::R
	nsteps::Int
end

edge(i,j) = Edge(min(j,i), max(j,i))

VGraphModel{V,M,R} =  GraphModel{V,E,M,R} where {V,E<:Nothing,M,R} 

Flux.@functor GraphModel

function vertexmap(g::GraphNode) 
	nv(g) == 0 && return(Vector{Int}())
	reduce(vcat, map(i -> g.fadjacency[i], 1:nv(g)))
end

function edgemap(g::GraphNode)
	nv(g) == 0 && return(Vector{Int}())
	reduce(vcat, map(i -> map(j -> g.edge2id[edge(i,j)], g.fadjacency.bags[i]), 1:nv(g)))
end

function millbags(g::GraphNode)
	length2bags(map(length, g.fadjacency.bags))
end

Zygote.@nograd vertexmap
Zygote.@nograd edgemap
Zygote.@nograd millbags


function (m::GraphModel)(g::GraphNode)
	o = messagepass(m, 
		g, 
		m.vlifter(g.vprops), 
		m.elifter(g.eprops), 
		m.nsteps)
	m.reducer(BagNode(o, g.components))
end

function (m::VGraphModel)(g::GraphNode)
	o = messagepass(m, 
		g, 
		m.vlifter(g.vprops),
		nothing,
		m.nsteps)
	m.reducer(BagNode(o, g.components))
end

function messagepass(m::GraphModel, g::GraphNode, vx, ex, n)
	n == 0 && return(vx)
	ds = message(g, vx, ex)
	messagepass(m, g, m.messager(ds), ex, n - 1)
end

#####
#	message
#####

"""
	message(g::SimpleGraph{Int64}, mg)

	Constructs the `BagNode` implementing the message passing algorithm
"""
function message(mg, vertex_properties::AbstractNode, edge_properties::AbstractNode)
	#construct the BagNode for a single message passing step 
	BagNode(
		ProductNode((
			vertex = vertex_properties[vertexmap(mg)],
			edge = edge_properties[edgemap(mg)],
			)),
		millbags(mg)
		)
end

function message(mg, vertex_properties::AbstractNode, edge_properties::Nothing)
	#construct the BagNode for a single message passing step 
	BagNode(vertex_properties[vertexmap(mg)], 
		millbags(mg)
		)
end


"""
	reflectinmodel(g::GraphNode; lastfnn, fnn, agg, lastfnn)

	constructs GNN model types of graphs provided in 
	`metagraph` with edge and vertex properties converted to Mill by 
	`extractor`. The Layers are constructing according to primitives described 
	in `fnn` and `agg`

	`lastfnn` is a function constructing the very last layer controlling the dimension of the output
		(a mandatory parameter).
		
	`metagraph` - an example of to be processed graph used to construct the layers. 
		The graph can be simple (two nodes with one edge), but it should have 
		all information on edges
	`extractor` is an extractor converting description of edges and vertices to Mill structures
	`fnn` is a function creating a feed forward neural network, default `fnn = d -> dense(d, 32,relu)`
	`agg` is an aggregation function projecting arbitrary number of Vectors to a single Vector `SegmentedMeanMax`
"""
function reflectinmodel(g::GraphNode; 
		lastfnn,
		fnn = d -> Dense(d, 64, relu), 
		agg = meanmax_aggregation,
		nsteps = 3,
		)
	vlifter = reflectinmodel(g.vprops, fnn, agg)
	elifter = isnothing(g.eprops) ? nothing : reflectinmodel(g.eprops, fnn, agg)
	lvx = vlifter(g.vprops)
	lex = isnothing(g.eprops) ? nothing : elifter(g.eprops)
	msg = message(g, lvx, lex)
	messager = reflectinmodel(msg, fnn, agg)
	onev = messager(msg)
	reducer = reflectinmodel(BagNode(onev, g.components), fnn, agg, fsm = Dict("" => lastfnn))
	GraphModel(vlifter, elifter, messager, reducer, nsteps)
end

#####
#	Hooking to HierarchicalUtils
#####

NodeType(::Type{<:GraphModel}) = InnerNode()
noderepr(::GraphModel) = "gnn"

children(n::GraphModel) = [:vertex => n.vlifter, :edge => n.elifter, :messager => n.messager, :reducer => n.reducer]

Base.show(io::IO, ::MIME"text/plain", @nospecialize(n::GraphModel)) = 
HierarchicalUtils.printtree(io, n; htrunc=3)
