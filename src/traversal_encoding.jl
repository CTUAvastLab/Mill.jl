_segment_width(l::Integer) = ceil(Int, log2(l+1))
encode(c::AbstractString, i::Integer, l::Integer) = c * string(i, base=2, pad=_segment_width(l))
decode(c::AbstractString, l::Integer) = (parse(Int, c[1:_segment_width(l)], base=2), c[1+_segment_width(l):end])

const ALPHABET = [Char(x) for x in vcat(collect.([48:57, 65:90, 97:122, 63:63, 33:33])...)]
const INV_ALPHABET = Dict(ALPHABET[i] => i for i in 1:length(ALPHABET))

function stringify(c::AbstractString)
    c = c * '0'^(6 - (length(c) % 6))
    join(ALPHABET[parse(Int, x, base=2) + 1] for x in [c[i:i+5] for i in 1:6:(length(c)-1)])
end

function destringify(c::AbstractString)
    join(string(INV_ALPHABET[x] - 1, base=2, pad=6) for x in c)
end

function _walk(m::Union{ArrayModel, ArrayNode}, c::AbstractString)
    i, _ = decode(c, length(c))
    return i == 0 ? m : error("Invalid index!")
end
    
function _walk(m::BagModel, c::AbstractString)
    i, nc = decode(c, 1)
    return i == 0 ? m : _walk(m.im, nc)
end

function _walk(n::AbstractBagNode, c::AbstractString)
    i, nc = decode(c, 1)
    return i == 0 ? n : _walk(n.data, nc)
end

function _walk(m::ProductModel, c::AbstractString)
    i, nc = decode(c, length(m.ms))
    return i == 0 ? m : _walk(m.ms[i], nc)
end

function _walk(n::AbstractTreeNode, c::AbstractString)
    i, nc = decode(c, length(n.data))
    0 <= i <= length(n.data) || error("Invalid index!")
    return i == 0 ? n : _walk(n.data[i], nc)
end

show_traversal(n::AbstractNode) = dsprint(Base.stdout, n, tr=true)
show_traversal(m::MillModel) = modelprint(Base.stdout, m, tr=true)
Base.getindex(n::AbstractNode, i::AbstractString) = _walk(n, destringify(i))
Base.getindex(m::MillModel, i::AbstractString) = _walk(m, destringify(i))
repr(s::AbstractString, traversal::Bool) = traversal ? " [$(stringify(s))]" : ""
