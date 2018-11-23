
"""
  BagChain(layers...)
	BagChain multiple layers / functions together, so that they are called in sequence
	on a given input supported by bags.

"""
struct BagChain{T<:Tuple}
  layers::T
  BagChain(xs...) = new{typeof(xs)}(xs)
end

Flux.@forward BagChain.layers Base.getindex, Base.first, Base.last, Base.lastindex
Flux.@forward BagChain.layers Base.iterate

children(c::BagChain) = c.layers
mapchildren(f, c::BagChain) = BagChain(f.(c.layers)...)

applychain(::Tuple{}, x, bags) = x
applychain(fs::Tuple, x, bags) = applychain(Base.tail(fs), first(fs)(x, bags), bags)

(c::BagChain)(x, bags) = applychain(c.layers, x, bags)
(c::BagChain)(x::ArrayNode, bags) = ArrayNode(applychain(c.layers, x.data, bags))
(c::BagChain)(x::BagNode) = applychain(c.layers, x.data, x.bags)

Base.getindex(c::BagChain, i::AbstractArray) = BagChain(c.layers[i]...)

Base.show(io::IO, m::BagChain) = modelprint(io, m)

function modelprint(io::IO, m::BagChain; pad=[])
    c = COLORS[(length(pad)%length(COLORS))+1]
    paddedprint(io, "BagChain (", color=c)
    for j in 1:length(m.layers) - 1
	    modelprint(io, m.layers[j])
	  end
    paddedprint(io, m.layers[end], "\n")
end


activations(c::BagChain, x, bags) = accumulate((x, m) -> m(x, bags), c.layers, init = x)
