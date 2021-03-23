using Mill, BenchmarkTools, Serialization, Zygote, Flux
using LinearAlgebra
using Setfield
using ThreadPools
using Profile, ProfileSVG
# ccall((:openblas_get_num_threads64_, Base.libblas_name), Cint, ())

deduplicatable(ds::ArrayNode) = true
deduplicatable(ds::BagNode) = false
function deduplicatable(ds::ProductNode)
	all(subds isa ArrayNode for subds in ds.data)
end

function dedupkey(ds::ProductNode, i)
	map(x -> dedupkey(x, i), ds.data)
end

function dedupkey(ds::ArrayNode{<:NGramMatrix,<:Any}, i)
	ds.data.s[i]
end

function dedupkey(ds::ArrayNode{<:Matrix,<:Any}, i)
	ds.data[:,i]
end

function dedupkey(ds::ArrayNode{Flux.OneHotMatrix{Array{Flux.OneHotVector,1}},<:Any}, i)
	ds.data.data[i].ix
end


function deduplicate(ds::BagNode)
	!deduplicatable(ds.data) && return(ds)

	keep = fill(false, Mill.nobs(ds.data))
	item2level = Dict{Any,Int}()
	idx2newidx = Dict{Any,Int}()
	newidx = 1
	for i in 1:Mill.nobs(ds.data)
		x = dedupkey(ds.data, i)
		if !haskey(item2level, x)
			item2level[x] = newidx
			newidx += 1
			keep[i] = true
		end
		idx2newidx[i] = item2level[x]
	end

	ds.data[keep]
	newbags = map(ds.bags) do b 
		isempty(b) && return(Vector{Int}())
		map(i -> idx2newidx[i], b)
	end
	BagNode(ds.data[keep],  Mill.ScatteredBags(newbags))
end

function deduplicate(dt::ProductNode)
	dt = @set dt.data.versionInfo = deduplicate(dt.data.versionInfo)
	dt = @set dt.data.sectionTable.data.sections = deduplicate(dt.data.sectionTable.data.sections)
	dt = @set dt.data.richHeader.data.richHeaderRecords = deduplicate(dt.data.richHeader.data.richHeaderRecords)
	dt = @set dt.data.resourceTable.data.resources = deduplicate(dt.data.resourceTable.data.resources)
	dt = @set dt.data.importTable.data.imports = deduplicate(dt.data.importTable.data.imports)
	dt = @set dt.data.exportTable.data.exports = deduplicate(dt.data.exportTable.data.exports)
	dt = @set dt.data.dotnetInfo.data.classes.data.data.methods = deduplicate(dt.data.dotnetInfo.data.classes.data.data.methods)
	dt = @set dt.data.certificateTable.data.certificates = deduplicate(dt.data.certificateTable.data.certificates)
end

m, dss = deserialize("threadtest/model_and_samples.jls")
ds = catobs(dss...);
dds = deduplicate(ds);
m(ds).data ≈ m(dds).data
ps = Flux.params(m);
@btime m(ds)								#
@btime m(ds)								#  
@btime m(dds)								#
@btime m(dds)								#  
@btime gradient(() -> sum(m(ds).data), ps)  #  
@btime gradient(() -> sum(m(ds).data), ps)  #
@btime gradient(() -> sum(m(dds).data), ps)  #  
@btime gradient(() -> sum(m(dds).data), ps)  #

Profile.clear()
@profile gradient(() -> sum(m(ds).data), ps)
ProfileSVG.save("/tmp/profile.svg")


# Debugging slow matmul
B = ds[:importTable][:imports].data[:libraryName]
m = ArrayModel(Chain(Dense(2053,32,relu), Dense(32,32)))
ps = Flux.params(m)
Profile.clear()
@profile gradient(() -> sum(m(B).data), ps)
ProfileSVG.save("/tmp/profile.svg")

W = randn(Float32,64,64)
X = randn(Float32,64,640000)
Profile.clear()
@profile gradient(W -> sum(sin.(W * X)), W)
ProfileSVG.save("/tmp/profile.svg")


######
# Non-threaded version on single thread
######
using Mill, BenchmarkTools, Serialization, Zygote, Flux
using LinearAlgebra
using Setfield
using PrayTools
m, dss = deserialize("model_and_samples.jls")
dss = tuple(dss...)
ds = catobs(dss...);
ps = Flux.params(m);
@btime m(ds)								
@btime m(ds)								
@btime gradient(() -> sum(m(ds).data), ps)  
@btime gradient(() -> sum(m(ds).data), ps)  

function Base.reduce(::typeof(Mill.catobs), as::Vector{T}) where {T <: Flux.OneHotMatrix{Array{Flux.OneHotVector,1}}}
    data = reduce(vcat, map(x -> x.data, as))
    Flux.OneHotMatrix(as[1].height, data)
end
@btime gradient(() -> sum(m(ds).data), ps)  


# Simulating Karel's approach to model paralelism
pdss = map(x -> (x,), dss)
@btime PrayTools.pgradient(x -> sum(m(x).data), ps, pdss)
