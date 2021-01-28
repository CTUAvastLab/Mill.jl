"""
		sparsify(x,nnzrate)

		replace matrices with at most `nnzrate` fraction of non-zeros with SparseMatrixCSC

```wip_jldoctest
julia> x = ProductNode((
				ProductNode((
					MatrixNode(randn(5,5)),
					MatrixNode(zeros(5,5))
						)),
				MatrixNode(zeros(5,5))
				))
julia> mapdata(i -> sparsify(i,0.05),x)

```
"""
sparsify(x,nnzrate) = x
sparsify(x::Matrix,nnzrate) = (mean(x .!= 0) <nnzrate) ? sparse(x) : x

# can be removed when https://github.com/FluxML/Flux.jl/pull/1357 is merged
function Base.:*(A::AbstractMatrix, B::Adjoint{Bool,<: Flux.OneHotMatrix})
    m = size(A,1)
    Y = similar(A, m, size(B,2))
    Y .= 0
    BT = B'
    for (j,ohv) in enumerate(BT.data)
        ix = ohv.ix
        for i in 1:m
            @inbounds Y[i,ix] += A[i,j]
        end
    end
    Y
end

pred_lens(n, p::Function) = _pred_lens(n, p)
list_lens(n) = pred_lens(n, t -> true)
findnonempty_lens(n) = pred_lens(n, t -> t isa AbstractNode && nobs(t) > 0)
find_lens(n, x) = pred_lens(n, t -> t === x)

_pred_lens(n, p::Function) = p(n) ? [IdentityLens()] : Lens[]
function _pred_lens(n::T, p::Function) where T <: MillStruct
    res = vcat([map(l -> PropertyLens{k}() ∘ l, _pred_lens(getproperty(n, k), p))
                for k in fieldnames(T)]...)
    p(n) ? [IdentityLens(); res] : res
end
function _pred_lens(n::Union{Tuple, NamedTuple}, p::Function)
    vcat([map(l -> IndexLens(tuple(i)) ∘ l, _pred_lens(n[i], p)) for i in eachindex(n)]...)
end

code2lens(n::MillStruct, c::AbstractString) = find_lens(n, n[c])
lens2code(n::MillStruct, l::Lens) = HierarchicalUtils.find_traversal(n, get(n, l))

function model_lens(model, lens::ComposedLens)
    outerlens = model_lens(model, lens.outer)
    outerlens ∘ model_lens(get(model, outerlens), lens.inner)
end
model_lens(::ArrayModel, ::PropertyLens{:data}) = @lens _.m
model_lens(::BagModel, ::PropertyLens{:data}) = @lens _.im
model_lens(::ProductModel, ::PropertyLens{:data}) = @lens _.ms
model_lens(::Union{NamedTuple, Tuple}, lens::IndexLens) = lens
model_lens(::Union{AbstractMillModel, NamedTuple, Tuple}, lens::IdentityLens) = lens

function data_lens(ds, lens::ComposedLens)
    outerlens = data_lens(ds, lens.outer)
    outerlens ∘ data_lens(get(ds, outerlens), lens.inner)
end
data_lens(::ArrayNode, ::PropertyLens{:m}) = @lens _.data
data_lens(::AbstractBagNode, ::PropertyLens{:im}) = @lens _.data
data_lens(::AbstractProductNode, ::PropertyLens{:ms}) = @lens _.data
data_lens(::Union{NamedTuple, Tuple}, lens::IndexLens) = lens
data_lens(::Union{AbstractNode, NamedTuple, Tuple}, lens::IdentityLens) = lens

replacein(x, oldnode, newnode) = x
replacein(x::Tuple, oldnode, newnode) = tuple([replacein(m, oldnode, newnode) for m in x]...)
replacein(x::NamedTuple, oldnode, newnode) = (;[k => replacein(x[k], oldnode, newnode) for k in keys(x)]...)

function replacein(x::T, oldnode, newnode) where T <: MillStruct
    x === oldnode && return(newnode)
    fields = map(f -> replacein(getproperty(x, f), oldnode, newnode), fieldnames(T))
    n = nameof(T)
    p = parentmodule(T)
    eval(:($p.$n))(fields...)
end

function replacein(x::LazyNode{N}, oldnode, newnode) where {N}
    x === oldnode && return newnode
    LazyNode{N}(replacein(x.data, oldnode, newnode))
end

function replacein(x::LazyModel{N}, oldnode, newnode) where {N}
    x === oldnode && return newnode
    LazyModel{N}(replacein(x.m, oldnode, newnode))
end
