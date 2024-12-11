import HierarchicalUtils: NodeType, LeafNode, InnerNode, children, nodeshow, nodecommshow

@nospecialize

NodeType(::Type{<:Union{Missing, ArrayNode, ArrayModel, LazyNode}}) = LeafNode()
NodeType(::Type{<:AbstractMillNode}) = InnerNode()
NodeType(::Type{<:AbstractMillModel}) = InnerNode()
NodeType(::Type{<:LazyModel}) = InnerNode()

children(n::AbstractBagNode) = (n.data,)
children(n::BagModel) = (n.im,)
children(n::ProductNode) = n.data
children(n::ProductModel) = n.ms
children(n::LazyModel) = (n.m,)

@specialize


"""
    datasummary(n::AbstractMillNode)

Print summary of parameters of node `n`.

# Examples
```jldoctest
julia> n = ProductNode(ArrayNode(randn(2, 3)))
ProductNode  3 obs
  ╰── ArrayNode(2×3 Array with Float64 elements)  3 obs

julia> datasummary(n)
"Data summary: 3 obs, 104 bytes."
```

See also: [`modelsummary`](@ref).
"""
function datasummary(n::AbstractMillNode)
    bytes = Base.format_bytes(Base.summarysize(n))
    string("Data summary: ", numobs(n), " obs, ", bytes, ".")
end

function Base.show(io::IO, ::MIME"text/plain", @nospecialize(n::AbstractMillNode))
    HierarchicalUtils.printtree(io, n; htrunc=5, vtrunc=10, breakline=false)
end

function Base.show(io::IO, ::MIME"text/plain", @nospecialize(n::ArrayNode))
    print(io, join(size(n), "×"), " ", summary(n))
    if !isempty(n.data)
        print(io, ":\n")
        Base.print_array(IOContext(io, :typeinfo => eltype(n.data)), n.data)
    end
end

function Base.show(io::IO, m::MIME"text/plain", @nospecialize(n::LazyNode))
    print(io, summary(n))
    if !isempty(n.data)
        print(io, ":\n")
        if n.data isa AbstractArray
            Base.print_array(IOContext(io, :typeinfo => eltype(n.data)), n.data)
        else
            print(io, " ")
            show(io, m, n.data)
        end
    end
end

nodeshow(io::IO, ::Missing) = print(io, "∅")
nodeshow(io::IO, ::LazyNode{N, Nothing}) where {N} = print(io, "LazyNode{$N} ∅")

nodecommshow(io::IO, @nospecialize(n::AbstractMillNode)) = print(io, " ", numobs(n), " obs")

function Base.show(io::IO, @nospecialize(n::AbstractMillNode))
    print(io, nameof(typeof(n)))
    get(io, :compact, false) || _show_data(io, n)
end

function Base.show(io::IO, @nospecialize(n::LazyNode{T})) where T
    print(io, nameof(typeof(n)), "{", T, "}")
    get(io, :compact, false) || _show_data(io, n)
end

_show_data(_, _) = nothing

function _show_data(io, n::ArrayNode{T}) where {T <: AbstractArray}
    print(io, "(")
    if ndims(n.data) == 1
        print(io, nameof(T), " of length ", length(n.data))
    else
        print(io, join(size(n), "×"), " ", nameof(T))
    end
    print(io, " with ", eltype(n.data), " elements)")
end

_show_data(io, n::LazyNode{N}) where {N} = print(io, "(", eltype(n.data), ")")

"""
    modelsummary(m::AbstractMillModel)

Print summary of parameters of model `m`.

# Examples
```jldoctest
julia> m = ProductModel(ArrayModel(Dense(2, 3)))
ProductModel ↦ identity
  ╰── ArrayModel(Dense(2 => 3))  2 arrays, 9 params, 124 bytes

julia> modelsummary(m)
"Model summary: 2 arrays, 9 params, 124 bytes"
```

See also: [`datasummary`](@ref).
"""
function modelsummary(m::AbstractMillModel)
    # params summary from https://github.com/FluxML/Flux.jl/blob/master/src/layers/show.jl
    ps = Flux.trainables(m)
    npars = Flux.underscorise(sum(length, ps))
    bytes = Base.format_bytes(sum(Base.summarysize, ps))
    string("Model summary: ", length(ps), " arrays, ", npars, " params, ", bytes)
end

function Base.show(io::IO, ::MIME"text/plain", @nospecialize(m::AbstractMillModel))
    HierarchicalUtils.printtree(io, m; htrunc=5, vtrunc=10, breakline = false)
end

_levelparams(m::ArrayModel) = Flux.params(m.m)
_levelparams(m::BagModel) = Flux.params(m.a, m.bm)
_levelparams(m::ProductModel) = Flux.params(m.m)
_levelparams(::LazyModel) = Flux.Params([])
_levelparams(m) = _levelparams(NodeType(m), m)
_levelparams(::LeafNode, m) = Flux.params(m)
function _levelparams(_, m)
    error("Define custom HierarchicalUtils.nodecommshow or Mill._levelparams for $(nameof(typeof(m)))")
end

function nodecommshow(io::IO, @nospecialize(m::AbstractMillModel))
    # params summary from https://github.com/FluxML/Flux.jl/blob/master/src/layers/show.jl

    # destructuralize params of special matrices
    destruct(m::AbstractArray{<:Number}) = [m]
    destruct(m) = isempty(m) ? collect(m) : reduce(vcat, map(destruct, collect(m)))
    destruct(m::ImputingMatrix) = [m.W, m.ψ]

    ps = _levelparams(m).params |> destruct
    if !isempty(ps)
        npars = Flux.underscorise(sum(length, ps))
        print(io, " ", length(ps), " arrays, ", npars, " params")
        if !isempty(ps) && Flux._all(iszero, ps)
            print(io, " (all zero)")
        elseif Flux._any(isnan, ps)
            print(io, " (some NaN)")
        elseif Flux._any(isinf, ps)
            print(io, " (some Inf)")
        end
        bytes = Base.format_bytes(sum(Base.summarysize, ps))
        print(io, ", ", bytes)
    end
end

function Base.show(io::IO, @nospecialize(n::AbstractMillModel))
    print(io, nameof(typeof(n)))
    get(io, :compact, false) || _show_submodels(io, n)
end

_show_submodels(io, m::ArrayModel) = print(io, "(", m.m, ")")
_show_submodels(io, m::BagModel) = print(io, " ↦ ", m.a, " ↦ ", m.bm)
_show_submodels(io, m::ProductModel) = print(io, " ↦ ", m.m)
_show_submodels(io, ::LazyModel{Name}) where {Name} = print(io, "{", Name, "}")
_show_submodels(io, _) = print(io)

function Base.showarg(io::IO, x::MaybeHotVector, toplevel)
    print(io, "MaybeHotVector")
    toplevel && print(io, " with eltype $(eltype(x))")
    return nothing
end

function Base.showarg(io::IO, X::MaybeHotMatrix, toplevel)
    print(io, "MaybeHotMatrix")
    toplevel && print(io, " with eltype $(eltype(X))")
    return nothing
end

# this is from /LinearAlgebra/src/diagonal.jl, official way to print the dots:
function Base.replace_in_print_matrix(x::MaybeHotArray, i::Integer, j::Integer, s::AbstractString)
    ismissing(x[i, j]) || x[i, j] ? s : Base.replace_with_centered_mark(s)
end

function Base.show(io::IO, X::T) where T <: Union{ImputingMatrix, MaybeHotArray, NGramMatrix}
    if get(io, :compact, false)
        if ndims(X) == 1
            print(io, length(X), "-element ", nameof(T))
        else
            print(io, join(size(X), "×"), " ", nameof(T))
        end
    else
        _show_fields(io, X)
    end
end

Base.show(io::IO, ::MIME"text/plain", a::AbstractAggregation) = _show_fields(io, a)

function _show_fields(io, x::T; context = :compact => true) where {T}
    o = join(["$f = $(repr(getfield(x, f); context))" for f in fieldnames(T)], ", ")
    print(io, nameof(T), "(", o, ")")
end

function Base.print_array(io::IO, A::ImputingMatrix)
    println(io, "W:")
    Base.print_array(io, A.W)
    println(io, "\n\nψ:")
    _print_params(io, A)
end

_print_params(io::IO, A::PreImputingMatrix) = Base.print_array(io, A.ψ |> permutedims)
_print_params(io::IO, A::PostImputingMatrix) = Base.print_array(io, A.ψ)

Base.print_array(io::IO, A::NGramMatrix) = Base.print_array(io, A.S)

function Base.show(io::IO, l::Dense{F,<:ImputingMatrix}) where {F}
    print(io, _name(l.weight), "Dense(", size(l.weight, 2), " => ", size(l.weight, 1))
    l.σ == identity || print(io, ", ", l.σ)
    l.bias == false && print(io, "; bias=false")
    print(io, ")")
end

function Base.show(io::IO, ::MIME"text/plain", l::Dense{F,<:ImputingMatrix}) where {F}
    show(io, l)
    if !get(io, :compact, false)
        print(io, ' ')
        nodecommshow(io, ArrayModel(l))
    end
end

_name(::PreImputingMatrix) = "[preimputing]"
_name(::PostImputingMatrix) = "[postimputing]"

function Base.show(io::IO, a::T) where T <: AbstractAggregation
    print(io, nameof(T))
    get(io, :compact, false) || print(io, "(", length(a), ")")
end
