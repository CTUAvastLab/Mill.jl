const _emptyismissing = Ref(false)

"""
Returns the current value of the `emptyismissing` parameter. See the [Empty bags](@ref) section of the docs for more information.
"""
emptyismissing() = _emptyismissing[]

"""
Sets the new value of the `emptyismissing` parameter. See the [Empty bags](@ref) section of the docs for more information.
"""
emptyismissing!(a) = _emptyismissing[] = a


const _bagcount = Ref(true)

"""
Returns the current value of the `bagcount` parameter. See the [Bag count](@ref) section of the docs for more information.
"""
bagcount() = _bagcount[]

"""
Sets the new value of the `bagcount` parameter. See the [Bag count](@ref) section of the docs for more information.
"""
bagcount!(a) = _bagcount[] = a


const _string_start_code = Ref(0x02) # STX in ascii

"""
    Mill.string_start_code()

Returns the current value of the `string_start_code` parameter used as a code point of the abstract string-start character. The default value of the parameter is `0x02`, which corresponds to the `STX` character in ASCII encoding. See the [Strings](@ref) section of the docs for more information.
"""
string_start_code() = _string_start_code[]

"""
    Mill.string_start_code!(c::UInt8)

Sets the new value of the `string_start_code` parameter used as a code point of the abstract string-start character to `c`. The default value of the parameter is `0x02`, which corresponds to the `STX` character in ASCII encoding. See the [Strings](@ref) section of the docs for more information.
"""
string_start_code!(c) = _string_start_code[] = c


const _string_end_code = Ref(0x03) # ETX in ascii

"""
    Mill.string_end_code()

Returns the current value of the `string_end_code` parameter used as a code point of the abstract string-end character. The default value of the parameter is `0x03`, which corresponds to the `ETX` character in ASCII encoding. See the [Strings](@ref) section of the docs for more information.
"""
string_end_code() = _string_end_code[]

"""
    Mill.string_end_code!(c::UInt8)

Sets the new value of the `string_end_code` parameter used as a code point of the abstract string-end character to `c`. The default value of the parameter is `0x03`, which corresponds to the `ETX` character in ASCII encoding. See the [Strings](@ref) section of the docs for more information.
"""
string_end_code!(c) = _string_end_code[] = c


const _wildcard_code = Ref(0x00) # NUL in ascii

"""
"""
wildcard_code() = _wildcard_code[]

"""
"""
wildcard_code!(c) = _wildcard_code[] = c
