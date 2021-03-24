using LightGraphs
using Mill
using Test
using Flux
using Mill: rem_vertices, rem_edges

include("testgraphs.jl")

@testset "catobs of graphs" begin
	for edgeprops in [true, false]
		g1 = create3graph(3,edgeprops)
		g2 = create3graph(2,edgeprops)
		gg = catobs(g1, g2)
		@test gg.vprops[:a].data[:,1:3] ≈ g1.vprops[:a].data
		@test gg.vprops[:a].data[:,4:6] ≈ g2.vprops[:a].data
		@test gg.vprops[:b].data.data[:,1:5] ≈ g1.vprops[:b].data.data
		@test gg.vprops[:b].data.data[:,6:10] ≈ g2.vprops[:b].data.data
		if edgeprops
			@test gg.eprops.data[:,1:3] ≈ g1.eprops.data
			@test gg.eprops.data[:,4:5] ≈ g2.eprops.data
		end
		@test gg.vprops[:b].bags.bags ==  UnitRange{Int64}[1:2, 3:5, 0:-1, 6:7, 8:10, 0:-1]
		@test gg.fadjacency.bags ==  [[2, 3], [1, 3], [1, 2], [5], [4,6], [5]]
		@test gg.fadjacency.bags[1:3] ==  g1.fadjacency.bags
		@test gg.fadjacency.bags[4:end] ==  map(x -> x .+ 3, g2.fadjacency.bags)
		@test gg.components.bags ==  [1:3, 4:6]
		@test gg.edge2id[Edge(1,2)] == 1
		@test gg.edge2id[Edge(1,3)] == 2
		@test gg.edge2id[Edge(2,3)] == 3
		@test gg.edge2id[Edge(4,5)] == 4
		@test gg.edge2id[Edge(5,6)] == 5

		# Let's try opposite order
		gg = catobs(g2, g1)
		@test gg.vprops[:a].data[:,1:3] ≈ g2.vprops[:a].data
		@test gg.vprops[:a].data[:,4:6] ≈ g1.vprops[:a].data
		@test gg.vprops[:b].data.data[:,1:5] ≈ g2.vprops[:b].data.data
		@test gg.vprops[:b].data.data[:,6:10] ≈ g1.vprops[:b].data.data
		if edgeprops
			@test gg.eprops.data[:,1:2] ≈ g2.eprops.data
			@test gg.eprops.data[:,3:5] ≈ g1.eprops.data
		end
		@test gg.vprops[:b].bags.bags ==  UnitRange{Int64}[1:2, 3:5, 0:-1, 6:7, 8:10, 0:-1]
		@test gg.fadjacency.bags ==  [[2], [1, 3], [2], [5, 6], [4, 6], [4, 5]]
		@test gg.fadjacency.bags[1:3] ==  g2.fadjacency.bags
		@test gg.fadjacency.bags[4:end] ==  map(x -> x .+ 3, g1.fadjacency.bags)
		@test gg.components.bags ==  [1:3, 4:6]
		@test gg.edge2id[Edge(1,2)] == 1
		@test gg.edge2id[Edge(2,3)] == 2
		@test gg.edge2id[Edge(4,5)] == 3
		@test gg.edge2id[Edge(4,6)] == 4
		@test gg.edge2id[Edge(5,6)] == 5
	end


	for edgeprops in [true, false]
		g1 = create3graph(1, edgeprops)
		g2 = create3graph(1, edgeprops)
		gg = catobs(g1, g2)
		@test gg.vprops[:a].data[:,1:3] ≈ g1.vprops[:a].data
		@test gg.vprops[:a].data[:,4:6] ≈ g2.vprops[:a].data
		@test gg.vprops[:b].data.data[:,1:5] ≈ g1.vprops[:b].data.data
		@test gg.vprops[:b].data.data[:,6:10] ≈ g2.vprops[:b].data.data
		if edgeprops
			@test gg.eprops.data[:,1:1] ≈ g1.eprops.data
			@test gg.eprops.data[:,2:2] ≈ g2.eprops.data
		end
		@test gg.vprops[:b].bags.bags ==  UnitRange{Int64}[1:2, 3:5, 0:-1, 6:7, 8:10, 0:-1]
		@test gg.fadjacency.bags ==   Vector{Vector{Int64}}([[2], [1], [], [5], [4], []])
		@test gg.fadjacency.bags[1:3] ==  g1.fadjacency.bags
		@test gg.fadjacency.bags[4:end] ==  map(x -> x .+ 3, g2.fadjacency.bags)
		@test gg.components.bags ==  [1:3, 4:6]
		@test gg.edge2id[Edge(1,2)] == 1
		@test gg.edge2id[Edge(4,5)] == 2
	end
end

@testset "getindex of graph components" begin
	for edgeprops in [true, false]
		g1 = create3graph(3, edgeprops)
		g2 = create3graph(2, edgeprops)
		gg = catobs(g1, g2)[2]
		@test gg.vprops[:a].data ≈ g2.vprops[:a].data
		@test gg.vprops[:b].data.data ≈ g2.vprops[:b].data.data
		@test gg.vprops[:b].bags.bags == g2.vprops[:b].bags.bags
		@test gg.eprops == g2.eprops
		@test gg.fadjacency.bags ==  g2.fadjacency.bags
		@test gg.components.bags ==  g2.components.bags
		@test gg.edge2id == g2.edge2id

		gg = catobs(g2, g1)[2]
		@test gg.vprops[:a].data ≈ g1.vprops[:a].data
		@test gg.vprops[:b].data.data ≈ g1.vprops[:b].data.data
		@test gg.vprops[:b].bags.bags == g1.vprops[:b].bags.bags
		@test gg.eprops == g1.eprops
		@test gg.fadjacency.bags ==  g1.fadjacency.bags
		@test gg.components.bags ==  g1.components.bags
		@test gg.edge2id == g1.edge2id

		gs = graphsuite()
		gg = reduce(catobs, gs)
		for i in length(gs)
			gi = gg[i]
			g1 = gs[i]
			@test gi.vprops[:a].data ≈ g1.vprops[:a].data
			@test gi.vprops[:b].data.data ≈ g1.vprops[:b].data.data
			@test gi.vprops[:b].bags.bags == g1.vprops[:b].bags.bags
			@test gi.eprops == g1.eprops
			@test gi.fadjacency.bags ==  g1.fadjacency.bags
			@test gi.components.bags ==  g1.components.bags
			@test gi.edge2id == g1.edge2id
		end
	end
end

@testset "testing removing vertices" begin
	for edgeprops in [true, false]
		g = create3graph(3, edgeprops)
		gg = rem_vertices(g, [1])	
		if edgeprops
			@test gg.eprops == g.eprops[[g.edge2id[Edge(2,3)]]]
		end
		@test gg.vprops[:a].data == g.vprops[2:3][:a].data
		@test gg.vprops[:b].data.data == g.vprops[2:3][:b].data.data
		@test gg.vprops[:b].bags.bags == g.vprops[2:3][:b].bags.bags
		@test gg.fadjacency.bags ==  [[2],[1]]
		@test gg.components.bags ==  [1:2]
		@test gg.edge2id[Edge(1,2)] == 1

		g = catobs(create3graph(1,edgeprops), create3graph(2,edgeprops), create3graph(3,edgeprops))
		gg = rem_vertices(g, [1,5,9])
		if edgeprops
			@test gg.eprops == g.eprops[[g.edge2id[Edge(7,8)]]]
		end
		@test gg.vprops[:a].data == g.vprops[[2,3,4,6,7,8]][:a].data
		@test gg.vprops[:b].data.data == g.vprops[[2,3,4,6,7,8]][:b].data.data
		@test gg.vprops[:b].bags.bags == g.vprops[[2,3,4,6,7,8]][:b].bags.bags
		@test gg.fadjacency.bags ==  [[], [], [], [], [6], [5]]
		@test gg.components.bags ==  [1:2, 3:4, 5:6]
		@test gg.edge2id[Edge(5,6)] == 1
	end
end

@testset "testing removing edges" begin
	for edgeprops in [true, false]
		g = create3graph(3, edgeprops)
		gg = rem_edges(g, [2])
		if edgeprops
			@test gg.eprops == g.eprops[[1,3]]
		end
		@test gg.vprops[:a].data == g.vprops[:a].data
		@test gg.vprops[:b].data.data == g.vprops[:b].data.data
		@test gg.vprops[:b].bags.bags == g.vprops[:b].bags.bags
		@test gg.fadjacency.bags ==  [[2], [1, 3], [2]]
		@test gg.components.bags ==  [1:3]
		@test gg.edge2id[Edge(1,2)] == 1
		@test gg.edge2id[Edge(2,3)] == 2

		g = catobs(create3graph(1, edgeprops), create3graph(2, edgeprops), create3graph(3, edgeprops))
		gg = rem_edges(g, [2,4,6])
		if edgeprops
			@test gg.eprops == g.eprops[[1,3,5]]
		end
		@test gg.vprops[:a].data == g.vprops[:a].data
		@test gg.vprops[:b].data.data == g.vprops[:b].data.data
		@test gg.vprops[:b].bags.bags == g.vprops[:b].bags.bags
		@test gg.fadjacency.bags ==  [[2], [1], [], [], [6], [5], [9], [], [7]]
		@test gg.components.bags ==  [1:3, 4:6, 7:9]
		@test gg.edge2id[Edge(1,2)] == 1
		@test gg.edge2id[Edge(5,6)] == 2
		@test gg.edge2id[Edge(7,9)] == 3
	end
end


@testset "Integration of GraphNode with GraphModel" begin
	for edgeprops in [true, false]
		m = reflectinmodel(create3graph(3, edgeprops), lastfnn = d -> Chain(Dense(d, 32), Dense(32, 10)))
		ds = graphsuite(edgeprops)
		@test mapreduce(x -> m(x).data, hcat,  ds) ≈ m(catobs(ds...)).data

		ds = catobs(ds...)
		ps = Flux.params(m)
		@testset "creation and gradient of the model" begin 
			@test length(ps) == length(Flux.params(m.elifter)) +
				length(Flux.params(m.vlifter)) + 
				length(Flux.params(m.messager)) +
				length(Flux.params(m.reducer))

			@test m(ds) isa ArrayNode{Matrix{Float32}, Nothing}

			gs = gradient(() -> sum(m(ds).data), ps)
			@test all(gs[p] ≠ nothing for p in ps)
		end
	end
end

