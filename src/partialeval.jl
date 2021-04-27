function partialeval(m::IdentityModel, ds::ArrayNode, skipnode)
    ds === skipnode && return(m, skipnode, true)
    (m, skipnode, false)
end

function partialeval(m::ArrayModel, ds::ArrayNode, skipnode)
    ds === skipnode && return(m, skipnode, true)
    (identity_model(), m(ds), false)
end

# (m::BagModel)(x::WeightedBagNode{<: AbstractMillNode}) = m.bm(m.a(m.im(x.data), x.bags, x.weights))

function partialeval(m::BagModel, ds::BagNode, skipnode)
    ds === skipnode && return(m, skipnode, true)
    im, ids, keep = partialeval(m.im, ds.data, skipnode)
    if keep
        return (BagModel(im, m.a,  m.bm), BagNode(ids, ds.bags, ds.metadata), true)
    end
    (identity_model(), m.bm(m.a(ids, ds.bags)), false)
end

function partialeval(m::BagModel, ds::WeightedBagNode, skipnode)
    ds === skipnode && return(m, skipnode, true)
    im, ids, keep = partialeval(m.im, ds.data, skipnode)
    if keep
        return (BagModel(im, m.a,  m.bm), WeightedBagNode(ids, ds.bags, ds.weights, ds.metadata),  true)
    end
    (identity_model(), m.bm(m.a(ids, ds.bags, ds.weights)), false)
end

function Mill.partialeval(m::LazyModel{N}, ds::LazyNode{N}, skipnode) where {N}
    ds === skipnode && return(m, skipnode, true)
    (identity_model(), m(ds), false)
end

function partialeval(m::ProductModel{M}, ds::ProductNode{P}, newnode) where {P<:NamedTuple, M<:NamedTuple} 
    ks = keys(m.ms)
    mods = map(ks) do k
        partialeval(m.ms[k], ds.data[k], newnode)
    end
    ms = map(f -> f[1], mods)
    dd = map(f -> f[2], mods)
    if any(f[3] for f in mods)
        return (ProductModel((;zip(ks, ms)...), m.m), ProductNode((;zip(ks, dd)...), ds.metadata), true)
    end
    (identity_model(), m.m(reduce(vcat, dd |> collect)), false)
end

function partialeval(m::ProductModel{M}, ds::ProductNode{P}, newnode) where {P<:Tuple, M<:Tuple} 
    mods = map(1:length(m.ms)) do k
        partialeval(m.ms[k], ds.data[k], newnode)
    end
    ms = map(f -> f[1], mods)
    dd = map(f -> f[2], mods)
    if any(f[3] for f in mods)
        return (ProductModel(tuple(ms...), m.m), ProductNode(tuple(dd...), ds.metadata), true)
    end
    (identity_model(), m.m(reduce(vcat, dd |> collect)), false)
end
