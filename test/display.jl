using OpenQuantumBase, Test

replstr(x, kv::Pair...) = sprint((io,x) -> show(IOContext(io, :limit => true, :displaysize => (24, 80), kv...), MIME("text/plain"), x), x)
showstr(x, kv::Pair...) = sprint((io,x) -> show(IOContext(io, :limit => true, :displaysize => (24, 80), kv...), x), x)

A = (s) -> (1 - s)
B = (s) -> s
u = [1.0 + 0.0im, 1] / sqrt(2)
ρ = u * u'

H = DenseHamiltonian([A, B], [σx, σz])
@test replstr(H) == "\e[36mDenseHamiltonian\e[0m with \e[36mComplex{Float64}\e[0m\nwith size: (2, 2)"