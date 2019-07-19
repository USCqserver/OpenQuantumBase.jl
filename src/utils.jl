macro CSI_str(str)
    return :(string("\x1b[", $(esc(str)), "m"))
end

const TYPE_COLOR = CSI"36"
const NO_COLOR = CSI"0"
