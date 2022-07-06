using OpenQuantumBase, Test

replstr(x, kv::Pair...) = sprint((io,x) -> show(IOContext(io, :limit => true, :displaysize => (24, 80), kv...), MIME("text/plain"), x), x)
# only test the repl string
#showstr(x, kv::Pair...) = sprint((io,x) -> show(IOContext(io, :limit => true, :displaysize => (24, 80), kv...), x), x)

A = (s) -> (1 - s)
B = (s) -> s
u = [1.0 + 0.0im, 1] / sqrt(2)
ρ = u * u'

H = DenseHamiltonian([A, B], [σx, σz])
@test replstr(H) == "\e[36mDenseHamiltonian\e[0m with \e[36mComplexF64\e[0m\nwith size: (2, 2)"

annealing = Annealing(H, u)
@test_broken replstr(annealing) == "\e[36mAnnealing\e[0m with \e[36mOpenQuantumBase.DenseHamiltonian{ComplexF64}\e[0m and u0 \e[36mVector{ComplexF64}\e[0m\nu0 size: (2,)"

coupling = ConstantCouplings(["Z"])
@test replstr(coupling) == "\e[36mConstantCouplings\e[0m with \e[36mComplexF64\e[0m\nand string representation: [\"Z\"]"
@test replstr(ConstantCouplings([σz])) == "\e[36mConstantCouplings\e[0m with \e[36mComplexF64\e[0m\nand string representation: nothing"

η = 0.01; W = 5; fc = 4; T = 12.5
@test replstr(HybridOhmic(W, η, fc, T)) == "Hybrid Ohmic bath instance:\nW (mK): 5.0\nϵl (GHz): 0.02083661222512523\nη (unitless): 0.01\nωc (GHz): 4.0\nT (mK): 12.5"

η = 1e-4; ωc = 4; T = 12
bath = Ohmic(η, ωc, T)
@test replstr(bath) == "Ohmic bath instance:\nη (unitless): 0.0001\nωc (GHz): 4.0\nT (mK): 12.0"

interaction = Interaction(coupling, bath)
@test replstr(interaction) == "\e[36mInteraction\e[0m with \e[36mConstantCouplings\e[0m with \e[36mComplexF64\e[0m\nand string representation: [\"Z\"]\nand bath: OpenQuantumBase.OhmicBath(0.0001, 25.132741228718345, 0.6365195925819416)"

iset = InteractionSet(interaction)
@test replstr(iset) == "\e[36mInteractionSet\e[0m with 1 interactions"

H = SparseHamiltonian([A, B], [spσx, spσz])
@test replstr(H) == "\e[36mSparseHamiltonian\e[0m with \e[36mComplexF64\e[0m\nwith size: (2, 2)"