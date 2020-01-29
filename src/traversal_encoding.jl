const ALPHABET = [Char(x) for x in vcat(collect.([42:43, 48:57, 65:90, 97:122])...)]
const INV_ALPHABET = Dict(c => i for (i,c) in enumerate(ALPHABET))

_segment_width(l::Integer) = ceil(Int, log2(l+1))
encode(i::Integer, l::Integer) = string(i, base=2, pad=_segment_width(l))
function decode(c::AbstractString, l::Integer)
    k = min(_segment_width(l), length(c))
    parse(Int, c[1:k], base=2), c[k+1:end]
end

function stringify(c::AbstractString)
    if length(c) % 6 != 0
        c = c * '0' ^ mod(6-length(c), 6)
    end
    join(ALPHABET[parse(Int, x, base=2) + 1] for x in [c[i:i+5] for i in 1:6:(length(c)-1)])
end

function destringify(c::AbstractString)
    join(string(INV_ALPHABET[x] - 1, base=2, pad=6) for x in c)
end

descendants(m::BagModel) = [m.im]
descendants(n::AbstractBagNode) = [n.data]
descendants(m::ProductModel) = m.ms
descendants(n::AbstractTreeNode) = n.data
descendants_n(m) = length(descendants(m))

function ith_child(m::T, i::Integer) where T
    try
        return descendants(m)[i]
    catch
        @error "Invalid index!"
    end
end

function _walk(m::Union{ArrayModel, ArrayNode}, c::AbstractString)
    !isempty(c) || return m
    i, nc = decode(c, length(c))
    if i == 0 && Set(nc) ⊆ ['0']
        return m
    else
        @error "Invalid index!"
    end
end
    
function _walk(m::Union{BagModel, AbstractBagNode}, c::AbstractString)
    !isempty(c) || return m
    i, nc = decode(c, 1)
    if i == 0
        if Set(nc) ⊆ ['0']
            return m
        else
            @error "Invalid index!"
        end
    end
    _walk(descendants(m)[1], nc)
end

function _walk(m::ProductModel, c::AbstractString)
    !isempty(c) || return m
    i, nc = decode(c, length(m.ms))
    if i == 0
        if Set(nc) ⊆ ['0']
            return m
        else
            @error "Invalid index!"
        end
    end
    _walk(m.ms[i], nc)
end

function _walk(n::AbstractTreeNode, c::AbstractString)
    !isempty(c) || return n
    i, nc = decode(c, length(n.data))
    0 <= i <= length(n.data) || @error "Invalid index!"
    if i == 0
        if Set(nc) ⊆ ['0']
            return n
        else
            @error "Invalid index!"
        end
    end
    _walk(n.data[i], nc)
end

function list_traversal(m::Union{ArrayNode, ArrayModel}, s::String="") 
    [stringify(s)]
end 

function list_traversal(m::Union{AbstractNode, MillModel}, s::String="")
    d = descendants(m)
    n = length(d)
    vcat(stringify(s), [list_traversal(d[i], s * encode(i, n)) for i in 1:n]...)
end 

show_traversal(n::AbstractNode) = dsprint(Base.stdout, n, tr=true)
show_traversal(m::MillModel) = modelprint(Base.stdout, m, tr=true)
Base.getindex(n::AbstractNode, i::AbstractString) = i == "" ? n : _walk(n, destringify(i))
Base.getindex(m::MillModel, i::AbstractString) = i == "" ? m : _walk(m, destringify(i))

tr_repr(s::AbstractString, traversal::Bool) = traversal ? " [$(stringify(s))]" : ""

encode_traversal(m::AbstractNode, idxs::Integer...) = stringify(_encode_traversal(m, idxs...))

function _encode_traversal(m, idxs...)
    !isempty(idxs) || return ""
    n = ith_child(m, idxs[1])
    return encode(idxs[1], descendants_n(m)) * _encode_traversal(n, idxs[2:end]...)
end
