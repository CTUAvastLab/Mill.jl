function transform(qs, ks, vs, bags)
	d = size(qs,1)
	os = map(bags) do b
		isempty(b) && return(vs[:,0:-1])
		q, k, v = qs[:,b], ks[:,b], vs[:,b]
		v * softmax(q' * k ./ sqrt(d))
	end
	reduce(catobs, os)
end

struct SingleHeadTransform{Q,K,V}
	q::Q 
	k::K 
	v::V
	n::Flux.LayerNorm
end

SingleHeadTransform(d) = SingleHeadTransform(Dense(d,d),Dense(d,d),Dense(d,d), LayerNorm(d))

Flux.@functor(SingleHeadTransform)

(m::SingleHeadTransform)(::Missing, bags) = missing

function (m::SingleHeadTransform)(x, bags)
	o = transform(m.q(x), m.k(x), m.v(x), bags)
	m.n(x + o)
end
