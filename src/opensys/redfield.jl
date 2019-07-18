struct Redfield <: AbstractOpenSys
    ops
    unitary
    cfun
end

function (R::Redfield)(du, u, p, t)
end
