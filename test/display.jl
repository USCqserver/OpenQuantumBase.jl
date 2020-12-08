using OpenQuantumBase, Test

replstr(x, kv::Pair...) = sprint((io,x) -> show(IOContext(io, :limit => true, :displaysize => (24, 80), kv...), MIME("text/plain"), x), x)
# only test the repl string
#showstr(x, kv::Pair...) = sprint((io,x) -> show(IOContext(io, :limit => true, :displaysize => (24, 80), kv...), x), x)

A = (s) -> (1 - s)
B = (s) -> s
u = [1.0 + 0.0im, 1] / sqrt(2)
ρ = u * u'

H = DenseHamiltonian([A, B], [σx, σz])
@test replstr(H) == "\e[36mDenseHamiltonian\e[0m with \e[36mComplex{Float64}\e[0m\nwith size: (2, 2)"
#@test showstr(H) == "\e[36mDenseHamiltonian\e[0m with \e[36mComplex{Float64}\e[0m\nwith size: (2, 2)"

H = SparseHamiltonian([A, B], [spσx, spσz])
@test replstr(H) == "\e[36mSparseHamiltonian\e[0m with \e[36mComplex{Float64}\e[0m\nwith size: (2, 2)"
#@test showstr(H) == "\e[36mSparseHamiltonian\e[0m with \e[36mComplex{Float64}\e[0m\nwith size: (2, 2)"

coupling = ConstantCouplings(["Z"])
@test replstr(coupling) == "\e[36mConstantCouplings\e[0m with \e[36mComplex{Float64}\e[0m\nand string representation: [\"Z\"]"
@test replstr(ConstantCouplings([σz])) == "\e[36mConstantCouplings\e[0m with \e[36mComplex{Float64}\e[0m\nand string representation: nothing"