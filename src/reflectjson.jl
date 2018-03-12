struct DataEntry{A,B}
	name::A 
	default::B
	transform::Function
end

struct DataSchema{A}
	items::A
end

DataSchema(a...) = DataSchema(tuple(a...)) 


reflect(s::DataSchema,d::Dict) = map(e -> reflect(e,get(d,e.name,nothing)),s.items)
reflect(s::E,d::B) where {E<:DataEntry,B<:Array} = s.transform(d) 
reflect(s::E,::Void) where {E<:DataEntry} = s.default

represent(a::Vector) = Ragged(reshape(a,:,1),nothing)
represent(a::Matrix) = Ragged(a,[1:size(a,2)])

DataEntry(a,b) = DataEntry(a,represent(b),represent)