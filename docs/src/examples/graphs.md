## GNNs with Mill in 16 lines

As has been mentioned in [Simon], the multi-instance learning is a essential piece to implement message passing in inference over graphs (called spatial Graph Neural Networks). It is really simple to make the idea fly is really small.

Let's assume a graph `g`, in this case created by `barabasi_albert` function, and let's assume that each vertex is described by a feature matrix `x`
```julia
using LightGraphs, Mill, Flux
g = barabasi_albert(10, 3, 2)
x = ArrayNode(randn(Float32, 7, 10))
```

What we do, is that we use `Mill.ScatteredBags` from `Mill.jl` to encode the neighbors of each vertex. That means that each vertex will be described by a bag of its neighbors. This information is by convenience stored in `fadjlist` of a graph `g`, therefore the bags are constructed as
```julia
b = Mill.ScatteredBags(g.fadjlist)
```

Finally, we create two models. First pre-process the description of vertices to some latent dimension for message passing, we will call this `lift`, and then a network realizing the message passing, we will call this one `mp`
```julia
lift = reflectinmodel(x, d -> Dense(d, 10), d -> SegmentedMean(d))
xx = lift(x)
mp = reflectinmodel(BagNode(xx, b), d -> Dense(d, 10), d -> SegmentedMean(d))
```

Notice that `BagNode(xx, b)` now essentially encodes the features about vertices and also the adjacency matrix. This also means that one step of message passing algorithm can be realized as `mp(BagNode(xx,b))` and it is differentiable, which can be verified by executing `gradient(() -> sum(sin.(mp(BagNode(xx,b)).data)), Flux.params(mp))`. 

So if we put everything together, the GNN implementation is following block of code (16 lines of mostly sugar).
```julia
using Flux, Mill, LightGraphs, Statistics

struct GNN{L,M, R}
	lift::L
	mp::M
	m::R
end

Flux.@functor GNN

function mpstep(m::GNN, xx::ArrayNode, bags, n)
	n == 0 && return(xx)
	mpstep(m, m.mp(BagNode(xx, bags)), bags, n - 1)
end

function (m::GNN)(g, x, n)
	xx = m.lift(x)
	bags = Mill.ScatteredBags(g.fadjlist)
	o = mpstep(m, xx, bags, n)
	m.m(vcat(mean(o.data, dims = 2), maximum(o.data, dims = 2)))
end
```

The initialization of the model is little tedious, but defining two helper functions for creating feed-forward neural networks `ffnn` and aggregation `agg` helps a bit. On the end, the graph neural network is properly integrated with Flux ecosystem and suports automatic differentiation.
```julia
zdim = 10
rounds = 5

ffnn(d) = Chain(Dense(d, zdim, relu), Dense(zdim, zdim))
agg(d) = SegmentedMeanMax(d)

g = barabasi_albert(10, 3, 2)
x = ArrayNode(randn(Float32, 7, 10))
gnn = GNN(reflectinmodel(x, ffnn, agg),
	BagModel(ffnn(zdim), agg(zdim), ffnn(2zdim)),
	ffnn(2zdim)
	)

gnn(g, x, rounds)
gradient(() -> sum(sin.(gnn(g, x, rounds))), Flux.params(gnn))
```

The above implementation is surprisingly general, as it supports a rich description of vertices, by which we mean that the description can be anything expressible by Mill (full JSONs).
The missing piece is putting weights on edges, which would be a bit more complicated.