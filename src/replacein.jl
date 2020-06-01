using Setfield
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