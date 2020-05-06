struct LazyModel{Name,T,F} <: AbstractMillModel
    m::T
    extract::F
end
LazyModel{Name}(m::M, e::F) where {Name, M, F} = LazyModel{Name,M, F}(m, e)

Flux.@functor LazyModel

function (m::LazyModel{Name,T,F})(x::LazyNode{Name,D}) where {Name, T, F, D}
	ds = m.extract(x.data)
	m.m(ds)
end

function _reflectinmodel(ds::LazyNode{Name,T}, db, da, b, a, s) where {Name, T}
	extract = lazyextractfun(Val(Name))
    pm, d = Mill._reflectinmodel(extract(ds.data), db, da, b, a, s * Mill.encode(1, 1))
	LazyModel{Name}(pm, extract), d
end

function lazyextractfun end

noderepr(n::LazyModel{Name,T,F}) where {Name, T, F} = "Lazy$(Name)"
NodeType(::LazyModel) = LeafNode()
