```@setup mill 
using Mill
using StatsBase: nobs
```

# Hierarchical utils
Mill.jl uses [HierarchicalUtils.jl](https://github.com/Sheemon7/HierarchicalUtils.jl) which brings a lot of additional features.

```@example mill 
using HierarchicalUtils
```

## Printing 

For instance, `Base.show` with `text/plain` MIME calls `HierarchicalUtils.printtree`:

```@repl mill
ds = BagNode(ProductNode((BagNode(ArrayNode(randn(4, 10)),
                                  [1:2, 3:4, 5:5, 6:7, 8:10]),
                          ArrayNode(randn(3, 5)),
                          BagNode(BagNode(ArrayNode(randn(2, 30)),
                                          [i:i+1 for i in 1:2:30]),
                                  [1:3, 4:6, 7:9, 10:12, 13:15]),
                          ArrayNode(randn(2, 5)))),
             [1:1, 2:3, 4:5])
printtree(ds; htrunc=3)
```

This can be used to print a non-truncated version of a model:

```@repl mill
printtree(ds)
```

## Traversal encoding

Callling with `trav=true` enables convenient traversal functionality with string indexing:

```@repl mill
m = reflectinmodel(ds)
printtree(m; trav=true)
```

This way any node in the model tree is swiftly accessible, which may come in handy when inspecting model parameters or simply deleting/replacing/inserting nodes to tree (for instance when constructing adversarial samples). All tree nodes are accessible by indexing with the traversal code:.

```@repl mill
m["Y"]
```

The following two approaches give the same result:

```@repl mill
m["Y"] === m.im.ms[1]
```

## Counting functions

Other functions provided by `HierarchicalUtils.jl`:

```@repl mill
nnodes(ds)
nleafs(ds)
NodeIterator(ds) |> collect
NodeIterator(ds, m) |> collect
LeafIterator(ds) |> collect
TypeIterator(BagModel, m) |> collect
PredicateIterator(x -> nobs(x) ≥ 10, ds) |> collect
```

For the complete showcase of possibilites, refer to [HierarchicalUtils.jl](https://github.com/Sheemon7/HierarchicalUtils.jl) and [this notebook](https://github.com/Sheemon7/HierarchicalUtils.jl/blob/master/examples/mill_integration.ipynb)
