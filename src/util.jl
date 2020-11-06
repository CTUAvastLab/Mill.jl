"""
		sparsify(x,nnzrate)

		replace matrices with at most `nnzrate` fraction of non-zeros with SparseMatrixCSC

```juliadoctest
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

findnonempty(ds::Union{ArrayNode, LazyNode}) = nobs(ds) == 0 ? nothing : [@lens _.data]
function findnonempty(ds::BagNode)
    childlenses = findnonempty(ds.data)
    isnothing(childlenses) ? childlenses : map(l -> Setfield.PropertyLens{:data}() ∘ l, childlenses)
end
function findnonempty(ds::ProductNode)
    lenses = mapreduce(vcat, keys(ds)) do k 
        childlenses = findnonempty(ds[k])
        isnothing(childlenses) && return childlenses
        map(l -> Setfield.PropertyLens{:data}() ∘ (Setfield.PropertyLens{k}() ∘ l), childlenses)
    end
    isnothing(lenses) && return nothing
    lenses = filter(!isnothing, lenses)
    isempty(lenses) ? nothing : lenses
end
function ModelLens(model::ProductModel, lens::Setfield.ComposedLens)
    if lens.outer == @lens _.data
        return(Setfield.PropertyLens{:ms}() ∘ ModelLens(model.ms, lens.inner))
    end
    return lens
end

function ModelLens(model, lens::Setfield.ComposedLens)
    outerlens = ModelLens(model, lens.outer)
    outerlens ∘ ModelLens(get(model, outerlens), lens.inner)
end
ModelLens(::BagModel, lens::Setfield.PropertyLens{:data}) = @lens _.im
ModelLens(::NamedTuple, lens::Setfield.PropertyLens) = lens
ModelLens(::ProductModel, lens::Setfield.PropertyLens{:data}) = @lens _.ms
ModelLens(::ArrayModel, ::Setfield.PropertyLens{:data}) = @lens _.m

replacein(x, oldnode, newnode) = x
replacein(x::Tuple, oldnode, newnode) = tuple([replacein(m, oldnode, newnode) for m in x]...)
replacein(x::NamedTuple, oldnode, newnode) = (;[k => replacein(x[k], oldnode, newnode) for k in keys(x)]...)

function replacein(x::T, oldnode, newnode) where {T<:Union{AbstractNode, AbstractMillModel}}
    x === oldnode && return(newnode)
    fields = map(f -> replacein(getproperty(x, f), oldnode, newnode), fieldnames(T))
    n = nameof(T)
    p = parentmodule(T)
    eval(:($p.$n))(fields...)
end

function replacein(x::LazyNode{N}, oldnode, newnode) where {N}
    x === oldnode && return(newnode)
    LazyNode{N}(replacein(x.data, oldnode, newnode))
end

function replacein(x::LazyModel{N}, oldnode, newnode) where {N}
    x === oldnode && return(newnode)
    LazyModel{N}(replacein(x.m, oldnode, newnode))
end

function findin(x, node)
    x === node && return(@lens _)
    return(nothing)
end

function findin(x::T, node) where {T<:Union{AbstractNode, AbstractMillModel}}
    x === node && return(@lens _)
    for k in fieldnames(T)
        l = findin(getproperty(x, k), node)
        if l != nothing
            lo = Setfield.PropertyLens{k}() ∘ l
            return(lo)
        end
    end
    return(nothing)
end

function findin(x::NamedTuple, node)
	x === node && return(@lens _)
    for k in keys(x)
    	l = findin(x[k], node)
    	if l != nothing
    		lo = Setfield.PropertyLens{k}() ∘ l
    		return(lo)
    	end
    end
    return(nothing)
end

function findin(x::Tuple, node)
    error("findin does not support Tuples due to restrinctions of Lens from Setfield.")
end

# function findin(x::T, node) where {T<:ArrayNode}
# 	x === node && return(@lens _)
# 	x.data === node && return(@lens _.data)
#     return(nothing)
# end

# function findin(x::LazyNode{N}, node) where {N}
# 	x === node && return(@lens _)
# 	x.data === node && return(@lens _.data)
#     return(nothing)
# end

# function findin(x::AbstractBagNode, node)
# 	x === node && return(@lens _)
# 	l = findin(x.data, node)
# 	if l != nothing 
# 		return((@lens _.data) ∘  l)
# 	end
#     return(nothing)
# end
