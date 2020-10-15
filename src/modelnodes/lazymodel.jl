struct LazyModel{Name,T} <: AbstractMillModel
    m::T
end

const LazyModel{Name} = LazyModel{Name, T} where {T}
LazyModel{Name}(m::M) where {Name, M} = LazyModel{Name,M}(m)

Flux.@functor LazyModel

function (m::LazyModel{Name})(x::LazyNode{Name}) where {Name}
    ds = unpack2mill(x)
    m.m(ds)
end

function _reflectinmodel(ds::LazyNode{Name}, db, da, b, a, s) where {Name}
    pm, d = Mill._reflectinmodel(unpack2mill(ds), db, da, b, a, s * Mill.encode(1, 1))
    LazyModel{Name}(pm), d
end

function unpack2mill end
