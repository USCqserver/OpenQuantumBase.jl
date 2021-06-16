"""
    cpvagk(f, t, a, b, tol=256*eps())

Calculate the Cauchy principle value integration of the form ``𝒫∫_a^b f(x)/(x-t) dx``. The algorithm is adapted from [P. Keller, 02.01.2015](https://www.sciencedirect.com/science/article/pii/S0377042715004422)
"""
function cpvagk(f, t, a, b, tol=256*eps())
    #   Adapted from P. Keller, 02.01.2015

    #   CPVAGK(F,T,A,B,TOL) = CPVInt[a,b]F(x)/(x-T)dx.
    #
    #   Based on QUADGK - the built-in Matlab adaptive integrator.
    #
    #   Q = CPVAGK(F,T,A,B,TOL) approximates the above integral
    #   and tries to make the absolute error less than TOL.
    #   The default value of TOL is 256*eps.
    #
    #   [Q,ERR] = CPVAGK(F,T,A,B,TOL) also provides the estimate of
    #   the absolute error. We suggest to always check the value of ERR.
    #   In case a QUADGK warning message appear, the approximate error bound
    #   provided inside the warning message should be ingnored!
    #   Consider only the value of ERR as error estimate.
    #
    #   Test (research) version.

    #   P. Keller, 02.01.2015


    if b<=t || t <= a
        error("CPVAGK: Invalid arguments")
    end
    s = min(1.0, t-a, b-t); #computing the split point(s)
    a = float(a)
    b = float(b)
    s = float(s)
    t = float(t)
    epsab = epsf = eps()
    ft = f(t)
    abt = 0.25*(abs(a) + abs(b) + 2*abs(t))
    tcond = 1 + 0.5*(max(1,abt)-1)^2/max(1,abt);
    #Estimating possible accuracy loss...
    # a) related to rounding t and interval endpoints:
    v = [a+2*epsf*abs(a), b-2*epsf*abs(b)];
    fd = max(max(1.0, abs(t)), abs.(v)...).*abs.(f.(v))./[t-a,b-t];
    erra = 0.75*epsab * sum(fd);
    # b) related to the magnitude of the 1st and 2nd derivative:
    sts = 255*s/256  # divided differences five steps...
    sth = min(sts, min(2,(b-a))./[82,70,32,22]...)
    ste = min(sts, exp(3/8*log(max(1,abs(t))*epsf)))
    dfs = max(abs(f(t+sts)-ft), abs(f(t-sts)-ft)) / sts  # <|f'[t-s,t+s]|
    dfh = max(abs(f(t+sth)-ft), abs(f(t-sth)-ft))./ sth./ [3/2, 7/4, 2, 3]
    df1 = abs(f(t+ste^2)-f(t-ste^2))/ste^2;  # ~~ 2*|f'(t)|
    df2 = sqrt(abs(f(t+ste)-2*ft+f(t-ste)))/ste;  # ~~ |f"(t)|^(1/2)
    errb1 = 17*tcond*epsf*(max(df1,dfh...) + abs(ft));
    errb2 = 10*tcond*epsf*df2;
    # c) related to the distance from interval endpoints:
    errc = 0.75*epsf*abt*abs(ft)/min(b-t,t-a)
    # Setting (safe) error tolerances, absolute and relative...
    atol = max(tol/2, 8*dfs*epsf, erra, errb1, errb2, errc);
    rtol = max(100*epsf, errb2, errc/max(sqrt(epsf), abs(ft)));
    # Setting parameters for quadgk()
    params = (atol=atol, rtol=rtol, maxevals=163840)
    # Computing the integrals...
    q1 = 0
    e1 = 0
    q2 = 0
    e2 = 0
    if s < t-a
        q1, e1 = quadgk((x)->(f(x)-ft)./(x-t), a, t-s; params...)
    end
    if s < b-t
        q2, e2 = quadgk((x)->(f(x)-ft)./(x-t), t+s, b; params...)
    end
    q3, e3 = quadgk((x)->(f(t+x)-f(t-x))./x, 0, s; params...)
    # Computing the final result and error estimation...
    q = q1 + q2 + q3 + ft*log((b-t)/(t-a))
    err = e1 + e2 + e3 + max(erra,errc) + errb1 + errb2 + 10*tcond*epsf*max(abs.([q1,q2,q3,q])...)
    q, err
end

"""
$(SIGNATURES)

Calculate the Lamb shift of spectrum `γ`. `atol` is the absolute tolerance for Cauchy principal value integral.
"""
function lambshift_cpvagk(w, γ; atol = 1e-7)
    g(x) = γ(x) / (x - w)
    cpv, cperr = cpvagk(γ, w, w - 1.0, w + 1.0)
    negv, negerr = quadgk(g, -Inf, w - 1.0)
    posv, poserr = quadgk(g, w + 1.0, Inf)
    v = cpv + negv + posv
    err = cperr + negerr + poserr
    if (err > atol) || (isnan(err))
        @warn "Absolute error of integration is larger than the tolerance."
    end
    -v / 2 / pi
end
# The above version of lamb shift calculation is used for test only

"""
$(SIGNATURES)

Calculate the Lamb shift of spectrum `γ` at angular frequency `ω`. All keyword arguments of `quadgk` function is supported.
"""
function lambshift(ω, γ; kwargs...)
    integrand = (x)->(γ(ω+x) - γ(ω-x))/x
    integral, = quadgk(integrand, 0, Inf; kwargs...)
    # TODO: do something with the error information
    -integral / 2 / π
end
