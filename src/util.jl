# can be removed when https://github.com/FluxML/OneHotArrays.jl/issues/32 is closed
function Base.reduce(::typeof(hcat), xs::Vector{T})  where T <: OneHotLike
    L = size(xs[1], 1)
    all(x -> size(x, 1) .== L, xs) ||
        throw(DimensionMismatch("The number of labels are not the same for all one-hot arrays."))
    OneHotArray(reduce(vcat, map(OneHotArrays._indices, xs)), L)
end

"""
    pred_lens(p, n)

Return a `Vector` of `Accessors.jl` lenses for accessing all nodes/fields in `n` conforming to
predicate `p`.

# Examples
```jldoctest
julia> n = ProductNode((BagNode(missing, bags([0:-1, 0:-1])), ArrayNode([1 2; 3 4])))
ProductNode  # 2 obs, 16 bytes
  ├── BagNode  # 2 obs, 88 bytes
  │     ╰── ∅
  ╰── ArrayNode(2×2 Array with Int64 elements)  # 2 obs, 80 bytes

julia> pred_lens(x -> x isa ArrayNode, n)
1-element Vector{Any}:
 (@optic _.data[2])
```

See also: [`list_lens`](@ref), [`find_lens`](@ref), [`findnonempty_lens`](@ref).
"""
function pred_lens(p::Function, n)
    result = Any[]
    _pred_lens!(p, n, (), result)
    return result
end

_pred_lens!(p::Function, x, l, result) = p(x) && push!(result, Accessors.opticcompose(l...))
function _pred_lens!(p::Function, n::T, l, result) where T <: AbstractMillStruct
    p(n) && push!(result, Accessors.opticcompose(l...))
    for k in fieldnames(T)
        _pred_lens!(p, getproperty(n, k), (l..., PropertyLens{k}()), result)
    end
end
function _pred_lens!(p::Function, n::Union{Tuple, NamedTuple}, l, result)
    for i in eachindex(n)
        _pred_lens!(p, n[i], (l..., IndexLens((i,))), result)
    end
end

"""
    list_lens(n)

Return a `Vector` of `Accessors.jl` lenses for accessing all nodes/fields in `n`.

# Examples
```jldoctest
julia> n = ProductNode((BagNode(missing, bags([0:-1, 0:-1])), ArrayNode([1 2; 3 4])))
ProductNode  # 2 obs, 16 bytes
  ├── BagNode  # 2 obs, 88 bytes
  │     ╰── ∅
  ╰── ArrayNode(2×2 Array with Int64 elements)  # 2 obs, 80 bytes

julia> list_lens(n)
9-element Vector{Any}:
 identity (generic function with 1 method)
 (@optic _.data[1])
 (@optic _.data[1].data)
 (@optic _.data[1].bags)
 (@optic _.data[1].metadata)
 (@optic _.data[2])
 (@optic _.data[2].data)
 (@optic _.data[2].metadata)
 (@optic _.metadata)
```

See also: [`pred_lens`](@ref), [`find_lens`](@ref), [`findnonempty_lens`](@ref).
"""
list_lens(n) = pred_lens(t -> true, n)

"""
    findnonempty_lens(n)

Return a `Vector` of `Accessors.jl` lenses for accessing all nodes/fields in `n` that contain at
least one observation.

# Examples
```jldoctest
julia> n = ProductNode((BagNode(missing, bags([0:-1, 0:-1])), ArrayNode([1 2; 3 4])))
ProductNode  # 2 obs, 16 bytes
  ├── BagNode  # 2 obs, 88 bytes
  │     ╰── ∅
  ╰── ArrayNode(2×2 Array with Int64 elements)  # 2 obs, 80 bytes

julia> findnonempty_lens(n)
3-element Vector{Any}:
 identity (generic function with 1 method)
 (@optic _.data[1])
 (@optic _.data[2])
```

See also: [`pred_lens`](@ref), [`list_lens`](@ref), [`find_lens`](@ref).
"""
findnonempty_lens(n) = pred_lens(t -> t isa AbstractMillNode && numobs(t) > 0, n)

"""
    find_lens(n, x)

Return a `Vector` of `Accessors.jl` lenses for accessing all nodes/fields in `n` that return `true`
when compared to `x` using `Base.===`.

# Examples
```jldoctest
julia> n = ProductNode((BagNode(missing, bags([0:-1, 0:-1])), ArrayNode([1 2; 3 4])))
ProductNode  # 2 obs, 16 bytes
  ├── BagNode  # 2 obs, 88 bytes
  │     ╰── ∅
  ╰── ArrayNode(2×2 Array with Int64 elements)  # 2 obs, 80 bytes

julia> find_lens(n, n.data[1])
1-element Vector{Any}:
 (@optic _.data[1])
```

See also: [`pred_lens`](@ref), [`list_lens`](@ref), [`findnonempty_lens`](@ref).
"""
find_lens(n, x) = pred_lens(t -> t ≡ x, n)

"""
    code2lens(n, c)

Convert code `c` from [HierarchicalUtils.jl](@ref) traversal to a `Vector` of `Accessors.jl`
lenses such that they access each node in tree `n` egal to node under code `c` in the tree.

# Examples
```jldoctest
julia> n = ProductNode((BagNode(missing, bags([0:-1, 0:-1])), ArrayNode([1 2; 3 4])));

julia> printtree(n; trav=true)
ProductNode [""]  # 2 obs, 16 bytes
  ├── BagNode ["E"]  # 2 obs, 88 bytes
  │     ╰── ∅ ["M"]
  ╰── ArrayNode(2×2 Array with Int64 elements) ["U"]  # 2 obs, 80 bytes

julia> code2lens(n, "U")
1-element Vector{Any}:
 (@optic _.data[2])
```

See also: [`lens2code`](@ref).
"""
code2lens(n::AbstractMillStruct, c::AbstractString) = find_lens(n, n[c])

"""
    lens2code(n, l)

Convert `Accessors.jl` lens `l` to a `Vector` of codes from [HierarchicalUtils.jl](@ref) traversal
such that they access each node in tree `n` egal to node accessible by lens `l`.

# Examples
```jldoctest
julia> n = ProductNode((BagNode(missing, bags([0:-1, 0:-1])), ArrayNode([1 2; 3 4])));

julia> printtree(n; trav=true)
ProductNode [""]  # 2 obs, 16 bytes
  ├── BagNode ["E"]  # 2 obs, 88 bytes
  │     ╰── ∅ ["M"]
  ╰── ArrayNode(2×2 Array with Int64 elements) ["U"]  # 2 obs, 80 bytes

julia> lens2code(n, (@optic _.data[2]))
1-element Vector{String}:
 "U"

julia> lens2code(n, (@optic _.data[∗]))
2-element Vector{String}:
 "E"
 "U"

```

See also: [`code2lens`](@ref).
"""
lens2code(n::AbstractMillStruct, l) = mapreduce(vcat, Accessors.getall(n, l)) do x
    HierarchicalUtils.find_traversal(n, x)
end

"""
    model_lens(m, l)

Convert `Accessors.jl` lens `l` for a data node to a new lens for accessing the same location in model `m`.

# Examples
```jldoctest
julia> n = ProductNode((BagNode(randn(Float32, 2, 2), bags([0:-1, 0:-1])),
                        ArrayNode(Float32[1 2; 3 4])))
ProductNode  # 2 obs, 24 bytes
  ├── BagNode  # 2 obs, 96 bytes
  │     ╰── ArrayNode(2×2 Array with Float32 elements)  # 2 obs, 64 bytes
  ╰── ArrayNode(2×2 Array with Float32 elements)  # 2 obs, 64 bytes

julia> m = reflectinmodel(n)
ProductModel ↦ Dense(20 => 10)  # 2 arrays, 210 params, 920 bytes
  ├── BagModel ↦ BagCount([SegmentedMean(10); SegmentedMax(10)]) ↦ Dense(21 => 10)  # 4 arrays, 240 params, 1.094 KiB
  │     ╰── ArrayModel(Dense(2 => 10))  # 2 arrays, 30 params, 200 bytes
  ╰── ArrayModel(Dense(2 => 10))  # 2 arrays, 30 params, 200 bytes

julia> model_lens(m, (@optic _.data[2]))
(@optic _.ms[2])
```

See also: [`data_lens`](@ref).
"""
function model_lens(model, lens::ComposedOptic)
    innerlens = model_lens(model, lens.inner)
    innerlens ⨟ model_lens(only(getall(model, innerlens)), lens.outer)
end
model_lens(::ArrayModel, ::PropertyLens{:data}) = @optic _.m
model_lens(::BagModel, ::PropertyLens{:data}) = @optic _.im
model_lens(::ProductModel, ::PropertyLens{:data}) = @optic _.ms
model_lens(::Union{NamedTuple, Tuple}, lens::IndexLens) = lens
model_lens(::Union{AbstractMillModel, NamedTuple, Tuple}, lens::typeof(identity)) = lens

"""
    data_lens(n, l)

Convert `Accessors.jl` lens `l` for a model node to a new lens for accessing the same location in data node `n`.

# Examples
```jldoctest
julia> n = ProductNode((BagNode(randn(Float32, 2, 2), bags([0:-1, 0:-1])), ArrayNode(Float32[1 2; 3 4])))
ProductNode  # 2 obs, 24 bytes
  ├── BagNode  # 2 obs, 96 bytes
  │     ╰── ArrayNode(2×2 Array with Float32 elements)  # 2 obs, 64 bytes
  ╰── ArrayNode(2×2 Array with Float32 elements)  # 2 obs, 64 bytes

julia> m = reflectinmodel(n)
ProductModel ↦ Dense(20 => 10)  # 2 arrays, 210 params, 920 bytes
  ├── BagModel ↦ BagCount([SegmentedMean(10); SegmentedMax(10)]) ↦ Dense(21 => 10)  # 4 arrays, 240 params, 1.094 KiB
  │     ╰── ArrayModel(Dense(2 => 10))  # 2 arrays, 30 params, 200 bytes
  ╰── ArrayModel(Dense(2 => 10))  # 2 arrays, 30 params, 200 bytes

julia> data_lens(n, (@optic _.ms[2]))
(@optic _.data[2])
```

See also: [`data_lens`](@ref).
"""
function data_lens(ds, lens::ComposedOptic)
    innerlens = data_lens(ds, lens.inner)
    innerlens ⨟ data_lens(only(getall(ds, innerlens)), lens.outer)
end
data_lens(::ArrayNode, ::PropertyLens{:m}) = @optic _.data
data_lens(::AbstractBagNode, ::PropertyLens{:im}) = @optic _.data
data_lens(::AbstractProductNode, ::PropertyLens{:ms}) = @optic _.data
data_lens(::Union{NamedTuple, Tuple}, lens::IndexLens) = lens
data_lens(::Union{AbstractMillNode, NamedTuple, Tuple}, lens::typeof(identity)) = lens

"""
    replacein(n, old, new)

Replace in data node or model `n` each occurence of `old` by `new`.

Short description

# Examples
```jldoctest
julia> n = ProductNode((BagNode(randn(2, 2), bags([0:-1, 0:-1])), ArrayNode([1 2; 3 4])))
ProductNode  # 2 obs, 24 bytes
  ├── BagNode  # 2 obs, 96 bytes
  │     ╰── ArrayNode(2×2 Array with Float64 elements)  # 2 obs, 80 bytes
  ╰── ArrayNode(2×2 Array with Int64 elements)  # 2 obs, 80 bytes

julia> replacein(n, n.data[1], ArrayNode(maybehotbatch([1, 2], 1:2)))
ProductNode  # 2 obs, 24 bytes
  ├── ArrayNode(2×2 MaybeHotMatrix with Bool elements)  # 2 obs, 80 bytes
  ╰── ArrayNode(2×2 Array with Int64 elements)  # 2 obs, 80 bytes
```
"""
replacein(x, _, _) = x
replacein(x::Tuple, oldnode, newnode) = tuple([replacein(m, oldnode, newnode) for m in x]...)
replacein(x::NamedTuple, oldnode, newnode) = (; [k => replacein(x[k], oldnode, newnode) for k in keys(x)]...)

function replacein(x::T, oldnode, newnode) where T <: AbstractMillStruct
    x ≡ oldnode && return(newnode)
    fields = map(f -> replacein(getproperty(x, f), oldnode, newnode), fieldnames(T))
    n = nameof(T)
    p = parentmodule(T)
    eval(:($p.$n))(fields...)
end

function replacein(x::LazyNode{N}, oldnode, newnode) where {N}
    x ≡ oldnode && return newnode
    LazyNode{N}(replacein(x.data, oldnode, newnode))
end

function replacein(x::LazyModel{N}, oldnode, newnode) where {N}
    x ≡ oldnode && return newnode
    LazyModel{N}(replacein(x.m, oldnode, newnode))
end
