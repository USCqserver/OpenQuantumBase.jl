using QTBase, Test
# test suite for Ohmic bath
η = 1e-4
ωc = 8 * pi
β = 1 / 2.23
bath = QTBase.OhmicBath(η, ωc, β)
@test correlation(0.02, 0.01, bath) == correlation(0.01, bath)
@test γ(0.0, bath) == 2 * pi * η / β
@test spectrum(0.0, bath) == 2 * pi * η / β
@test S(0.0, bath) ≈ -0.0025132734115775254 atol = 1e-6

# test suite for CustomBath
cfun = (t) -> exp(-t)
bath = CustomBath(correlation=cfun)
correlation(1, bath)
@test correlation(2, 1, bath) == correlation(1, bath)

# test suite for ensemble fluctuators
rtn = QTBase.SymetricRTN(2.0, 2.0)
@test 4 * exp(-2 * 3) == correlation(3, rtn)
@test 2 * 4 * 2 / (4 + 4) == spectrum(2, rtn)

ensemble_rtn = EnsembleFluctuator([1.0, 2.0], [2.0, 1.0])
@test exp(-2 * 3) + 4 * exp(-3) == correlation(3, ensemble_rtn)
@test 2 * 2 / (9 + 4) + 2 * 4 / (9 + 1) == spectrum(3, ensemble_rtn)

# test suite for HybridOhmic bath
η = 0.01; W = 5; fc = 4; T = 12.5
bath = HybridOhmic(W, η, fc, T)
@test S(0.0, bath) ≈ -0.2872777516270734
@test spectrum(0.0, bath) ≈ 1.7045312175373621

# test suite for correlated bath
coupling = ConstantCouplings([σ₊, σ₋], unit=:ħ)
γfun(w) = w >= 0 ? 1.0 : exp(-0.5)
cbath = CorrelatedBath(((1, 2), (2, 1)), spectrum=[(w) -> 0 γfun; γfun (w) -> 0])
γm = QTBase.build_spectrum(cbath)
@test γm[1, 1](0.0) == 0
@test γm[2, 2](0.0) == 0
@test γm[1, 2](0.5) == 1.0
@test γm[2, 1](0.5) == 1.0

D = QTBase.davies_from_interactions(InteractionSet(Interaction(coupling, cbath)), 1:10, false, nothing)[1]
du = zeros(ComplexF64, 2, 2)
ρ = [0.5 0;0 0.5]
ω = [1, 2]
ω =  ω' .- ω
D(du, ρ, ω, 0.5)
@test du ≈ [(1 - exp(-0.5)) * 0.5 0; 0 -(1 - exp(-0.5)) * 0.5]