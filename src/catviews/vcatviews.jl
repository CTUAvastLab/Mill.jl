struct VCatView{T,N} <: AbstractMatrix{T}
	matrices::NTuple{N, Matrix{T}}
	offsets::NTuple{N + 1, Int}
end

function VCatView(xs::NTuple{N,T}) where {N, T<:Matrix}
	offsets = (0, accumulate(+, map(x -> size(x,1), xs))...)
	VCatView(xs, offsets)
end


Base.size(x::VCatView) = (x.offsets[end], size(x.matrices[1],2))
Base.size(x::VCatView, i::Int) = i == 1 ? x.offsets[end] : size(x.matrices[1], 2)
Base.eltype(x::VCatView{T,N}) where {T,N} = T
band(b::VCatView, i) = b.offsets[i]+1:b.offsets[i+1]

function Base.getindex(x::VCatView, i, j)
	offset = 0 
	while x.offsets[offset+1] < i 
		offset+=1;
	end
	x.matrices[offset][i - x.offsets[offset], j]
end

import Base.*

function *(a::Matrix, b::VCatView)
	o = view(a,:,band(b, 1)) * b.matrices[1]
	for i in 2:length(b.matrices)
		o .+= view(a,:,band(b, i)) * b.matrices[i]
	end
	o
end

function *(a::Matrix, b::Adjoint{<:Any,<:VCatView}) 
	hcat(map(x -> a * x', b.parent.matrices)...)
end

function *(a::Matrix, b::Transpose{<:Any,<:VCatView}) 
	hcat(map(x -> a * x', b.parent.matrices)...)
end

@adjoint function *(a::Matrix, b::VCatView)
  return a * b, Δ-> begin 
  		matrices = map(1:length(b.matrices)) do i 
			view(a,:,band(b, i))' * Δ
		end
		(Δ * b', VCatView(tuple(matrices...), b.offsets))
	end
end