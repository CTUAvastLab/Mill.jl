replacein(x, oldnode, newnode) = x
replacein(x::Tuple, oldnode, newnode) = tuple([replacein(m, oldnode, newnode) for m in x]...)
replacein(x::NamedTuple, oldnode, newnode) = (;[k => replacein(x[k], oldnode, newnode) for k in keys(x)]...)
function replacein(x::T, oldnode, newnode) where {T<:Union{AbstractNode, AbstractMillModel}}
    x == oldnode && return(newnode)
    fields = map(f -> replacein(getproperty(x, f), oldnode, newnode), fieldnames(T))
    n = nameof(T)
    p = parentmodule(T)
    eval(:($p.$n))(fields...)
end

# TODO
# @generated function replacein(x::T, oldnode, newnode) where T <: Union{AbstractNode, AbstractMillModel}
#     return quote
#         x === oldnode && return newnode
#         fields = map(f -> replacein(getproperty(x, f), oldnode, newnode), $(fieldnames(T)))
#         # $(T)(fields...)
#     end
# end
