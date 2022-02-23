const _emptyismissing = Ref{Bool}(@load_preference("emptyismissing", false))

"""
    Mill.emptyismissing()

Get the current value of the `emptyismissing` parameter.

See also: [`Mill.emptyismissing!`](@ref).
"""
emptyismissing() = _emptyismissing[]

"""
    Mill.emptyismissing!(::Bool; persist=false)

Set the new value to the `emptyismissing` parameter.

Set `persist=true` to persist this setting between sessions.

See also: [`Mill.emptyismissing`](@ref).
"""
function emptyismissing!(a::Bool; persist=false)
    _emptyismissing[] = a
    if persist
        @set_preferences!("emptyismissing" => a)
    end
end

const _string_start_code = Ref{UInt8}(UInt8(@load_preference("string_start_code", 0x02))) # STX in ascii

"""
    Mill.string_start_code()

Get the current value of the `string_start_code` parameter used as a code point of the abstract string-start character.
The default value of the parameter is `0x02`, which corresponds to the `STX` character in ASCII encoding.

See also: [`Mill.string_start_code!`](@ref), [`Mill.string_end_code`](@ref), [`Mill.string_end_code!`](@ref).
"""
string_start_code() = _string_start_code[]

"""
    Mill.string_start_code!(c::Integer; persist=false)

Set the new value to the `string_start_code` parameter used as a code point of the abstract string-start character to `c`.
The default value of the parameter is `0x02`, which corresponds to the `STX` character in ASCII encoding.

`c` should fit into `UInt8`.

Set `persist=true` to persist this setting between sessions.

See also: [`Mill.string_start_code`](@ref), [`Mill.string_end_code`](@ref), [`Mill.string_end_code!`](@ref).
"""
function string_start_code!(c::Integer; persist=false)
    _string_start_code[] = UInt8(c)
    if persist
        @set_preferences!("string_start_code" => c)
    end
end

const _string_end_code = Ref{UInt8}(UInt8(@load_preference("string_end_code", 0x03))) # ETX in ascii

"""
    Mill.string_end_code()

Get the current value of the `string_end_code` parameter used as a code point of the abstract string-end character.
The default value of the parameter is `0x03`, which corresponds to the `ETX` character in ASCII encoding.

See also: [`Mill.string_end_code!`](@ref), [`Mill.string_start_code`](@ref), [`Mill.string_start_code!`](@ref).
"""
string_end_code() = _string_end_code[]

"""
    Mill.string_end_code!(c::Integer; persist=false)

Set the new value to the `string_end_code` parameter used as a code point of the abstract string-end character to `c`.
The default value of the parameter is `0x03`, which corresponds to the `ETX` character in ASCII encoding.

`c` should fit into `UInt8`.

Set `persist=true` to persist this setting between sessions.

See also: [`Mill.string_end_code`](@ref), [`Mill.string_start_code`](@ref), [`Mill.string_start_code!`](@ref).
"""
function string_end_code!(c::Integer; persist=false)
    _string_end_code[] = UInt8(c)
    if persist
        @set_preferences!("string_end_code" => c)
    end
end

# WILDCARDS NOT USED YET
# const _wildcard_code = Ref(0x00) # NUL in ascii
#
# wildcard_code() = _wildcard_code[]
# wildcard_code!(c) = _wildcard_code[] = c
