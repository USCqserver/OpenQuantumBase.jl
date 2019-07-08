function 𝒯integrand(func, t; args=())
    function integrand(x, f)
        y = 1.0
        x[1] = x[1]*t
        for i in 2:length(x)
            x[i] = x[i]*x[i-1]
            y = y*x[i-1]
        end
        f.=func(x, args...)*y*t
    end
end
