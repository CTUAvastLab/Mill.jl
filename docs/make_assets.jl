using Graphs, GraphRecipes, Plots, Random

Random.seed!(22)

g = SimpleGraph(9)
for e in [(1, 2), (1, 3), (1, 4),
          (2, 4), (2, 5),
          (3, 4), (3, 5), (3, 6), (3, 8),
          (4, 5), (4, 6), (4, 9),
          (5, 7), (5, 8),
          (6, 5), (6, 7), (6, 8),
          (7, 8),
          (8, 9)
]
    add_edge!(g, e...)
end

gp = graphplot(adjacency_matrix(g); linecolor = :darkgrey,
                                    nodecolor=:lightgrey,
                                    fontsize=11,
                                    markersize=0.2, nodeshape=:circle,
                                    background_color=:transparent,
                                    markercolor = range(colorant"white", stop=colorant"grey", length=nv(g)),
                                    names=1:nv(g)
                                    )
savefig(gp, joinpath(@__DIR__, "src", "assets", "graph.svg"))

Random.seed!(4)

g = SimpleDiGraph(8)
for e in [(1, 2), (1, 3), (2, 4), (2, 5), (3, 5),
          (3, 6), (5, 7), (6, 5), (6, 8), (7, 8)]
    add_edge!(g, e...)
end

gp = graphplot(adjacency_matrix(g); linecolor = :darkgrey,
                                    nodecolor=:lightgrey,
                                    fontsize=11,
                                    markersize=0.2, nodeshape=:circle,
                                    background_color=:transparent,
                                    names=range('a', length=nv(g))
                                    )

savefig(gp, joinpath(@__DIR__, "src", "assets", "dag.svg"))
