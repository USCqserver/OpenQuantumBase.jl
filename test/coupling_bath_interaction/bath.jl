using OpenQuantumBase, Test
# test suite for Ohmic bath
η = 1e-4
ωc = 8 * pi
β = 1 / 2.23
bath = OpenQuantumBase.OhmicBath(η, ωc, β)
cfun_test = OpenQuantumBase.build_correlation(bath)
γfun_test = OpenQuantumBase.build_spectrum(bath)

@test correlation(0.02, 0.01, bath) == correlation(0.01, bath)
@test cfun_test[1, 1](0.02, 0.01) == correlation(0.01, bath)
@test γ(0.0, bath) == 2 * pi * η / β
@test γfun_test(0.0) == 2 * pi * η / β
@test spectrum(0.0, bath) == 2 * pi * η / β
@test S(0.0, bath) ≈ OpenQuantumBase.lambshift_cpvagk(0.0, (x)->γ(x, bath)) atol = 1e-4 rtol = 1e-4
# the following test is kept as a consistency check
@test OpenQuantumBase.lambshift_cpvagk(0.0, (x)->γ(x, bath)) ≈ -0.0025132734115775254 rtol = 1e-4

η = 1e-4;fc = 4;T = 16
bath = Ohmic(η, fc, T)
τsb, err_τsb = τ_SB(bath)
τb, err_τb = τ_B(bath, 100, τsb)
@test τsb ≈ 284.61181493 atol = 1e-6 rtol = 1e-6
@test τb ≈ 0.07638653 atol = 1e-6 rtol = 1e-6
τc, err_τc = coarse_grain_timescale(bath, 100)
@test τc ≈ sqrt(τsb * τb / 5) atol = 1e-6 rtol = 1e-6

# test suite for CustomBath
cfun = (t) -> exp(-abs(t))
sfun = (ω) -> 2 / (1 + ω^2)
bath = CustomBath(correlation=cfun, spectrum=sfun)
@test correlation(1, bath) ≈ exp(-1)
@test correlation(2, 1, bath) == correlation(1, bath)
@test spectrum(0, bath) ≈ 2
@test γ(0, bath) ≈ 2
@test S(0, bath) == 0

# test suite for ensemble fluctuators
rtn = OpenQuantumBase.SymetricRTN(2.0, 2.0)
@test 4 * exp(-2 * 3) == correlation(3, rtn)
@test 2 * 4 * 2 / (4 + 4) == spectrum(2, rtn)

ensemble_rtn = EnsembleFluctuator([1.0, 2.0], [2.0, 1.0])
@test exp(-2 * 3) + 4 * exp(-3) == correlation(3, ensemble_rtn)
@test 2 * 2 / (9 + 4) + 2 * 4 / (9 + 1) == spectrum(3, ensemble_rtn)

# test suite for HybridOhmic bath
η = 0.01; W = 5; fc = 4; T = 12.5
bath = HybridOhmic(W, η, fc, T)
@test_broken spectrum(0.0, bath) ≈ 1.7045312175373621
@test S(0.0, bath) ≈ OpenQuantumBase.lambshift_cpvagk(0.0, (x)->γ(x, bath)) atol = 1e-4 rtol = 1e-4
# the following test is kept as a consistency check
@test_broken OpenQuantumBase.lambshift_cpvagk(0.0, (x)->γ(x, bath)) ≈ -0.2872777516270734

# test suite for correlated bath
coupling = ConstantCouplings([σ₊, σ₋], unit=:ħ)
γfun= (w) -> w >= 0 ? exp(-w) : exp(0.8w)
cbath = CorrelatedBath(((1, 2), (2, 1)), spectrum=[(w) -> 0 γfun; γfun (w) -> 0])
γm = OpenQuantumBase.build_spectrum(cbath)
@test γm[1, 1](0.0) == 0
@test γm[2, 2](0.0) == 0
@test γm[1, 2](0.5) == exp(-0.5)
@test γm[2, 1](-0.5) == exp(-0.5*0.8)

@test_throws ArgumentError OpenQuantumBase.build_correlation(cbath)
lambfun_1 = OpenQuantumBase.build_lambshift([0.0], false, cbath, Dict())
@test lambfun_1[1,1](0.0) == 0
@test lambfun_1[2,2](0.1) == 0
@test lambfun_1[1,2](0.5) == 0
@test lambfun_1[2,2](1.0) == 0

lambfun_2 = OpenQuantumBase.build_lambshift([], true, cbath, Dict())
lambfun_3 = OpenQuantumBase.build_lambshift(range(-5,5,length=1000), true, cbath, Dict())
lambfun_4 = OpenQuantumBase.build_lambshift([], true, cbath, Dict(:order=>1))
lambfun_5 = OpenQuantumBase.build_lambshift([0.0, 0.01], true, cbath, Dict(:order=>1))
@test isapprox(lambfun_2[1,2](0.0), 0.03551, atol=1e-4)
@test isapprox(lambfun_3[1,2](0.0), 0.03551, atol=1e-4)
@test lambfun_2[1,1](0) == 0
@test lambfun_3[2,2](0) == 0
@test lambfun_4[1,2](0) ≈ 0.03551 atol=1e-4
@test lambfun_5[2,1](0.01) ≈ 0.05019 atol=1e-4
