mutable struct Ragged{A}
	data::A 
	bags::Bags
end


"""
		function bag(k::Vector)


		create vector of unit ranges from keys k, assuming they are continuous 

```juliadoctest
julia> NestedMill.bag([2, 2, 2, 1, 1, 3])
3-element Array{UnitRange{Int64},1}:
 1:3
 4:5
 6:6
```

this will throw error ```NestedMill.bag([2, 2, 2, 1, 1, 3, 1])```
"""
function bag(k::Vector)
	b = Bags(length(unique(k)))
	i = 1
	bi = 1
	for j in 2:length(k)
		if k[j] != k[i]
			b[bi] = i:j-1 
			bi, i = bi+1,j 
		end 
	end 
	b[bi] = i:length(k)
	bi != length(b) && error("The number of bags should correspond to the number of unique items in k")
	b
end
