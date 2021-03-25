"""
	struct GraphNode{V<:AbstractNode,E,G<:LightGraphs.Edge} <: AbstractGraphNode
		vprops::V
		eprops::E
		edge2id::Dict{G, Int}
		fadjacency::ScatteredBags
		components::AlignedBags
	end


	`vprops <: AbstractNode` properties of vertices
	`eprops <: Union{AbstractNode,Nothing}` properties of edges
	`edge2id` maps `Edge` to an index of `eprops`. The property of an edge `e` is `eprops[edge2id[e]]`
	`fadjacency` is an adjacency matrix of a graph(s), for each vertex it contains a list (`Vector{Int}`)
			of indexes with which it is connected. No connection is indecated by empty vector.
	`components` identifies set of vertices of which single sample consists of. Although it is not enforced
			there should not be edges between vertices from different components.
"""
struct GraphNode{V<:AbstractNode,E,G<:LightGraphs.Edge} <: AbstractGraphNode
	vprops::V
	eprops::E
	edge2id::Dict{G, Int}
	fadjacency::ScatteredBags
	components::AlignedBags

	"""
		Description of graph has to posses some invariants that has to be enforced:
		* number of properties `eprops` has to be equal to the length of `edge2id`
		* number of edge descriptions in `fadjacency` has to be equal to the number 
		of vertices
	"""
	function GraphNode(vprops::V, eprops::E, edge2id::Dict{G, Int}, fadjacency, components) where {V,E,G<:LightGraphs.Edge}
		nobs(vprops) ≠ length(fadjacency) && error("adjacency list in `fadjacency` has to be equal to the number of vertices")
		eprops ≠ nothing && nobs(eprops) ≠ length(edge2id) && error("Number of edge properties has to be equal to the number of edges.")
		new{V,E,G}(vprops, eprops, edge2id, fadjacency, components)
	end
end

function Base.reduce(::typeof(catobs), as::Vector{T}) where {T<:GraphNode}
	vprops = [x.vprops for x in as]
	eprops = [x.eprops for x in as]
	components = _catbags([d.components for d in as])
	GraphNode(reduce(catobs, vprops),
		reduce(catobs, eprops),
		reduce_edge2id(as),
		reduce_adjacency(as),
		components
		)
end

"""
	reduce_adjacency(as::Vector{T}) where {T<:GraphNode}

	aggregate all edge2id to a single Dict while shifting 
	them by number of vertices of previous graphs
"""
function reduce_adjacency(as::Vector{T}) where {T<:GraphNode{<:Any,<:Any,G}} where {G<:LightGraphs.Edge}
	fadjacency = Vector{Vector{Int}}()
	v_offset = 0 
	for g in as 
		append!(fadjacency, map(x -> x .+ v_offset, g.fadjacency.bags))
		v_offset += nv(g)
	end
	ScatteredBags(fadjacency)
end

"""
	reduce_edge2id(as::Vector{T}) where {T<:GraphNode}

	aggregate all edge2id to a single Dict while shifting 
	them by number of vertices of previous graphs
"""
function reduce_edge2id(as::Vector{T}) where {T<:GraphNode{<:Any,<:Any,G}} where {G<:LightGraphs.Edge}
	edge2id = Dict{G, Int}()
	e_offset = 0 
	v_offset = 0 
	for g in as 
		for i in g.edge2id
			e = i.first
			edge2id[G(e.src + v_offset, e.dst + v_offset)] = i.second + e_offset
		end
		e_offset += ne(g)
		v_offset += nv(g)
	end
	edge2id
end


function Base.getindex(g::GraphNode, i::VecOrRange)
    components, v_kept = remapbags(g.components, i) #Mill 2.4

    v_removed = setdiff(1:nv(g), v_kept)
    emask, edge2id, fadj = remapedges_v(g.fadjacency, g.edge2id, v_removed) 
    GraphNode(g.vprops[v_kept],
		getprops(g.eprops, emask),
		edge2id,
		fadj,
		components
		)
end

catobs(as::GraphNode...) = reduce(catobs, collect(as))

Base.getindex(g::GraphNode, i::Int) = g[[i]]

"""
	rem_vertices(g::GraphNode, ii)

	remove a set of vertices from a graph
"""
function rem_vertices(g::GraphNode, ii::Vector{Int})
	keep_vertices = fill(true, nv(g))
	keep_vertices[ii] .= false
	components = adjustbags(g.components, keep_vertices)
    emask, edge2id, fadj = remapedges_v(g.fadjacency, g.edge2id, ii) 
    GraphNode(g.vprops[keep_vertices],
		getprops(g.eprops, emask),
		edge2id,
		fadj,
		components
		)
end

getprops(props::Nothing, mask) = nothing
getprops(props::AbstractNode, mask) = props[mask]

rem_vertices(g::GraphNode, m::Vector{Bool}) = rem_vertices(g, findall(m))
rem_vertices(g::GraphNode, m::BitVector) = rem_vertices(g, findall(m))

"""
	rem_edges(g::GraphNode, ii::Vector{Int})

	remove a set of edges from a graph
"""
function rem_edges(g::GraphNode, removed_edges::Vector{<:LightGraphs.Edge}) 
	emask, edge2id, fadj = remapedges_e(g.fadjacency, g.edge2id, removed_edges)
    GraphNode(g.vprops,
		getprops(g.eprops, emask),
		edge2id,
		fadj,
		g.components
		)
end

function rem_edges(g::GraphNode, removed_edges::Vector{Int})
	ii = collect(filter(e -> g.edge2id[e] ∈ removed_edges, keys(g.edge2id)))
	rem_edges(g, ii)
end

rem_edges(g::GraphNode, m::Vector{Bool}) = rem_edges(g, findall(m))
rem_edges(g::GraphNode, m::BitVector) = rem_edges(g, findall(m))


"""	
	kept_edges, edge2id, adjacency = remapedges_v(fadjacency, edge2id::Dict{D,Int}, ii::Vector{Int})
	
	remaps indexes of edges when vertices `ii` are removed

	performs three actions:
	1. changes numbers of vertices in `edge2id` 
	2. identify, which edges should be kept `preserved_edges`
	3. updates the adjacency list to keep edges only for a given vertices
"""
function remapedges_v(fadjacency::ScatteredBags, edge2id::Dict{D,Int}, removed_vertices::Vector{Int}) where {D}
	preserved_vertices = setdiff(1:length(fadjacency), removed_vertices)
	m = OrderedDict{Int, Int}((v => i for (i, v) in enumerate(preserved_vertices)))
	#identify, which edges should be preserved
	preserved_edges = fill(true, length(edge2id))
	new_edge2id = Dict{D,Int}()
	for (e, i) in edge2id
		if haskey(m, e.src) && haskey(m, e.dst) 
			new_edge2id[D(m[e.src], m[e.dst])] = i
		else
			preserved_edges[i] = false
		end
	end

	edgeorder = cumsum(preserved_edges)
	for (e, i) in new_edge2id
		new_edge2id[e] = edgeorder[i]
	end

	new_adjacency = map(fadjacency[preserved_vertices]) do jj 
		jj = filter(j -> haskey(m, j), jj)
		[m[i] for i in jj]
	end |> ScatteredBags
	preserved_edges, new_edge2id, new_adjacency
end

"""	
	preserved_edges, new_edge2id, new_adjacency = remapedges_e(fadjacency, edge2id, ii)
	
	remaps indexes of edges `ii` are removed

	preserved_edges --- which edges are present in the new graph
	new_edge2id --- new mapping of edges to indexes to their properties
	new_adjacency --- updated adjacency matrix
"""
function remapedges_e(fadjacency, edge2id, removed_edges::Vector{<:LightGraphs.Edge})
	preserved_indices = fill(true, length(edge2id))
	new_adjacency = deepcopy(fadjacency)
	for e in removed_edges
		i, j = e.src, e.dst
		new_adjacency[i] = filter(k -> k ≠ j, new_adjacency[i])
		new_adjacency[j] = filter(k -> k ≠ i, new_adjacency[j])
		preserved_indices[edge2id[e]] = false
	end

	edgeids = cumsum(preserved_indices)
	new_edge2id = map(collect(setdiff(keys(edge2id), removed_edges))) do e 
		e => edgeids[edge2id[e]]
	end |> Dict
	preserved_indices, new_edge2id, new_adjacency
end

function remapedges_e(fadjacency, edge2id, ii::Vector{Int})
	ii = collect(filter(e -> edge2id[e] ∈ ii, keys(edge2id)))
	remapedges_e(fadjacency, edge2id, ii)
end

StatsBase.nobs(g::GraphNode) = length(g.components)
LightGraphs.nv(g::GraphNode) = nobs(g.vprops)
LightGraphs.ne(g::GraphNode) = length(g.edge2id)

HierarchicalUtils.NodeType(::Type{<:GraphNode}) = InnerNode()
HierarchicalUtils.noderepr(::GraphNode) = "Graph"
HierarchicalUtils.children(n::GraphNode) = [:vertices => n.vprops, :edges => n.eprops]
HierarchicalUtils.children(n::GraphNode{V,E,G}) where {V<:AbstractNode,E<:Nothing,G<:LightGraphs.Edge} = [:vertices => n.vprops]
HierarchicalUtils.children(n::GraphNode) = [:vertices => n.vprops, :edges => n.eprops]
Base.show(io::IO, ::MIME"text/plain", @nospecialize(n::GraphNode)) = 
	HierarchicalUtils.printtree(io, n; htrunc=3)

import Base: ==
function ==(a::GraphNode, b::GraphNode)
	nv(a) != nv(b) && return(false)
	ne(a) != ne(b) && return(false)
	!(a.vprops == b.vprops) && return(false)
	!(a.eprops == b.eprops) && return(false)
	!(a.fadjacency.bags == b.fadjacency.bags) && return(false)
	!(a.edge2id == b.edge2id) && return(false)
	!(a.components == b.components) && return(false)
	return(true)
end