```@setup leafs
using Mill
```

# Data in leaves

In [`Mill.jl`](https://github.com/pevnak/Mill.jl) tree-like data representations, there are always some raw data on the leaf level, whereas on higher levels instances are grouped into bags ([`BagNode`](@ref)s), and different sets are joined together with Cartesion products ([`ProductNode`](@ref)s) and thus more abstract concepts are created. In this section we look into several examples how the lowest-level data can be represented.

For this purpose, let's assume that we would like to identify infected computers in a network from their HTTP traffic. Since one computer can make an arbitrary number of connections during the observation period, modelling the computer as a *bag* of connections seems like the most natural approach:

```@repl leafs
connections = AlignedBags([1:2, 3:3, 4:7, 8:8, 9:10])
```

Thus, each of the ten connections becomes an *instance* in one of the bags. How to represent this instance? Each HTTP flow has properties that can be expressed as standard numerical features, categorical features or strings of characters.

## Numerical features

We have already shown how to represent standard numerical features in previous parts of this manual. It is as simple as wrapping a type that behaves like a matrix into an [`ArrayNode`](@ref):

```@repl leafs
content_lengths = [4446, 1957, 4310, 11604, 17019, 13947, 13784, 15495, 3888, 11853]
dates = [1435420950, 1376190532, 1316869962, 1302775198, 1555598383,
         1562237892, 1473173059, 1325242539, 1508048391, 1532722821]
numerical_node = ArrayNode([content_lengths'; dates'])
```

We use `Content-Length`  and `Date` request headers, the latter converted to Unix timestamp.

## Categorical features

For categorical variables, we proceed in the same way, but we use one-hot encoding implemented in [`Flux.jl`](https://fluxml.ai). This way, we can encode for example a verb of the request:

```@repl leafs
using Flux

ALL_VERBS = ["GET", "HEAD", "POST", "PUT", "DELETE"] # etc...
verbs = ["GET", "GET", "POST", "HEAD", "HEAD", "HEAD", "HEAD", "PUT", "DELETE", "PUT"]
verb_node = ArrayNode(Flux.onehotbatch(verbs, ALL_VERBS))
```

or `Content-Encoding` header:

```@repl leafs
ALL_ENCODINGS = ["bzip2", "gzip", "xz", "identity"] # etc...
encodings = ["xz", "gzip", "bzip2", "xz", "identity", "bzip2", "identity", "identity", "xz", "xz"]
encoding_node = ArrayNode(Flux.onehotbatch(encodings, ALL_ENCODINGS))
```

Because `Flux.OneHotMatrix` supports multiplication it is possible to wrap it into an [`ArrayNode`](@ref).

## Strings

The last example we will consider are string features. This could for example be the `Host` header:

```@repl leafs
hosts = [
    "www.foo.com",
    "www.foo.com",
    "www.baz.com",
    "www.foo.com",
    "www.baz.com",
    "www.foo.com",
    "www.baz.com",
    "www.baz.com",
    "www.bar.com",
    "www.baz.com",
]
```

[`Mill.jl`](https://github.com/pevnak/Mill.jl) offers `n`gram histogram-based representation for strings. To get started, we pass the vector of strings into the constructor of [`NGramMatrix`](@ref):

```@repl leafs
hosts_ngrams = NGramMatrix(hosts, 3, 256, 7)
```

Each string gets processed into `n`grams (trigram in this case as specified in the first parameter). Then, each character is transformed into an integer via the `codeunits` function and the whole trigram is interpreted as a three digit number using a base `b` specified in the second parameter. Here, we use a base of `256`, which is the most reasonable choice for ascii URLs. For example, for `foo` trigram, we obtain:

```@repl leafs
c = codeunits("foo")
c[1] * 256^2 + c[2] * 256 + c[3]
```

The last step is taking the modulo of this result with respect to some prime modulo `m`, in this case `7` (last parameter in the constructor), leaving us with `3` as a result. Therefore, for this trigram `foo`, we would add `1` to the third row[^1]. We can convert this [`NGramMatrix`](@ref) into a sparse array and then to the standard array:

[^1]: One appropriate value for modulo `m` for real problems is `2053`

```@example leafs
using SparseArrays
```

```@repl leafs
hosts_dense = hosts_ngrams |> SparseMatrixCSC |> Matrix
```

Again, we get one column for each string, and the matrix has the same number of rows as modulo `m`. For each string `s`, we get `length(s) + n - 1` `n`grams:

```@repl leafs
sum(hosts_dense; dims=1)
```

This is because we use special abstract characters (or tokens) for the start and the end of the string. If we denote these `^` and `$`, respectively, from string `"foo"`, we get trigrams `^^f`, `^fo`, `foo`, `oo$`, `o$$`. Note that these special characters are purely abstract whereas `^` and `$` used only for illustration purposes here are characters like any other. Both string start and string end special characters have a unique mapping to integers, which can be obtained as well as set:

```@repl leafs
Mill.string_start_code()
Mill.string_end_code()
Mill.string_start_code!(42)
Mill.string_start_code()
```

[`NGramMatrix`](@ref) behaves like a matrix, implements an efficient left-multiplication and thus can be used in [`ArrayNode`](@ref):

```@repl leafs
hosts_ngrams::AbstractMatrix{Int64}
host_node = ArrayNode(hosts_ngrams)
```

[Adding custom nodes](@ref) section shows one more slightly more complex way of processing strings, specifically Unix paths.

## Putting it all together

Now, we can finally put wrap everything into one [`ProductNode`](@ref):

```@repl leafs
ds = ProductNode((
    numerical=numerical_node,
    verb=verb_node,
    encoding=encoding_node,
    hosts=host_node
))
```

create a model for training and compute some gradients:
```@repl leafs
m = reflectinmodel(ds)
gradient(() -> sum(Mill.data(m(ds))), params(m))
```

!!! ukn "Numerical features"
    To put all numerical features into one [`ArrayNode`](@ref) is a design choice. We could as well introduce more keys in the final [`ProductNode`](@ref). The model treats these two cases differently (see [Nodes](@ref) section).

This dummy example illustrates the versatility of [`Mill.jl`](https://github.com/pevnak/Mill.jl). With little to no preprocessing we are able to process complex hierarchical structures and avoid manually designing feature extraction procedures. For a more involved study on processing Internet traffic with [`Mill.jl`](https://github.com/pevnak/Mill.jl), see for example [Pevny2020](@cite).
