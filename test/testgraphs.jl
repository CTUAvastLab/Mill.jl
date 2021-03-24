function create3graph(ne = 3, edge_properties = true)
	g = LightGraphs.SimpleGraph(3)
	ne >= 1 && add_edge!(g, 1, 2)
	ne >= 2 && add_edge!(g, 2, 3)
	ne >= 3 && add_edge!(g, 1, 3)

	vx = ProductNode((a = ArrayNode(randn(2,3)),
		b = BagNode((ArrayNode(randn(2,5))), AlignedBags([1:2,3:5,0:-1]))
		))

	ex = edge_properties ? ArrayNode(randn(3,ne)) : nothing
	GraphNode(vx, 
		ex, 
		Dict(e => i	for (i,e) in enumerate(edges(g))),
		ScatteredBags(g.fadjlist),
		AlignedBags([1:3]))
end

function create0graph(edge_properties = true)
	vx = ProductNode((a = ArrayNode(randn(2,3)),
		b = BagNode((ArrayNode(randn(2,5))), AlignedBags([1:2,3:5,0:-1]))
		))

	ex = edge_properties ? ArrayNode(randn(3,0)) : nothing
	GraphNode(vx[0:-1], 
		ex, 
		Dict{LightGraphs.SimpleGraphs.SimpleEdge{Int64}, Int64}(),
		ScatteredBags{Int64}(Vector{Int64}[]),
		AlignedBags([0:-1]))
end


function create1graph(edge_properties = true)
	vx = ProductNode((a = ArrayNode(randn(2,3)),
		b = BagNode((ArrayNode(randn(2,5))), AlignedBags([1:2,3:5,0:-1]))
		))

	ex = edge_properties ? ArrayNode(randn(3,0)) : nothing
	GraphNode(vx[1:1], 
		ex, 
		Dict{LightGraphs.SimpleGraphs.SimpleEdge{Int64}, Int64}(),
		ScatteredBags{Int64}(Vector{Int64}[[]]),
		AlignedBags([1:1]))
end

function graphsuite(edge_properties = true)
	gs = [create3graph(mod(i,4), edge_properties) for i in 0:3]
	push!(gs, create1graph(edge_properties))
	push!(gs, create0graph(edge_properties))
	append!(gs, [create3graph(mod(i,4), edge_properties) for i in 0:3])
	push!(gs, create1graph(edge_properties))
	push!(gs, create0graph(edge_properties))
	gs
end


function createmetagraph()
	function randomvprops()
		Dict(:name => rand(["Adele", "Maria", "Karl", "Josef", "Gustav"]),
		:age => rand(1:100),
		:education => [rand(["Elementary", "Home"]), rand(["HighSchool", "Proffesional"]), rand(["Ms","Bsc"])])
	end

	function randomeprops()
		Dict(:relation => rand(["Love", "Hate", "Friend", "Fond", "Avoids"]))
	end
	
	g = MetaGraph(5)
	g = MetaGraph(barabasi_albert(10, 3, 2))
	foreach(i -> set_props!(g, i, randomvprops()), 1:nv(g.graph))
	foreach(e -> set_props!(g, e, randomeprops()), edges(g.graph))
	g
end
