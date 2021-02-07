const _emptyismissing = Ref(false)

"""
    Mill.emptyismissing()

Get the current value of the `emptyismissing` parameter.

See also: [`Mill.emptyismissing!`](@ref).
"""
emptyismissing() = _emptyismissing[]

"""
    Mill.emptyismissing!(::Bool)

Set the new value to the `emptyismissing` parameter.

See also: [`Mill.emptyismissing`](@ref).
"""
emptyismissing!(a) = _emptyismissing[] = a


const _bagcount = Ref(true)

"""
    Mill.bagcount()

Get the current value of the `bagcount` parameter.

See also: [`Mill.bagcount!`](@ref).
"""
bagcount() = _bagcount[]

"""
    Mill.bagcount!(Bool)

Set the new value to the `bagcount` parameter.

See also: [`Mill.bagcount`](@ref).
"""
bagcount!(a) = _bagcount[] = a


const _string_start_code = Ref(0x02) # STX in ascii

"""
    Mill.string_start_code()

Get the current value of the `string_start_code` parameter used as a code point of the abstract string-start character.
The default value of the parameter is `0x02`, which corresponds to the `STX` character in ASCII encoding.

See also: [`Mill.string_start_code!`](@ref), [`Mill.string_end_code`](@ref), [`Mill.string_end_code!`](@ref).
"""
string_start_code() = _string_start_code[]

"""
    Mill.string_start_code!(c::UInt8)

Set the new value to the `string_start_code` parameter used as a code point of the abstract string-start character to `c`.
The default value of the parameter is `0x02`, which corresponds to the `STX` character in ASCII encoding.

See also: [`Mill.string_start_code`](@ref), [`Mill.string_end_code`](@ref), [`Mill.string_end_code!`](@ref).
"""
string_start_code!(c) = _string_start_code[] = c


const _string_end_code = Ref(0x03) # ETX in ascii

"""
    Mill.string_end_code()

Get the current value of the `string_end_code` parameter used as a code point of the abstract string-end character.
The default value of the parameter is `0x03`, which corresponds to the `ETX` character in ASCII encoding.

See also: [`Mill.string_end_code!`](@ref), [`Mill.string_start_code`](@ref), [`Mill.string_start_code!`](@ref).
"""
string_end_code() = _string_end_code[]

"""
    Mill.string_end_code!(c::UInt8)

Set the new value to the `string_end_code` parameter used as a code point of the abstract string-end character to `c`.
The default value of the parameter is `0x03`, which corresponds to the `ETX` character in ASCII encoding.

See also: [`Mill.string_end_code`](@ref), [`Mill.string_start_code`](@ref), [`Mill.string_start_code!`](@ref).
"""
string_end_code!(c) = _string_end_code[] = c


const _wildcard_code = Ref(0x00) # NUL in ascii

"""
"""
wildcard_code() = _wildcard_code[]

"""
"""
wildcard_code!(c) = _wildcard_code[] = c
