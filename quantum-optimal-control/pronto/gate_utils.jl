using QuantumOptics


function gauss_envelope(t, tg, A)
    A * (exp(-8 * t * (t - tg) / tg^2) - 1)
end

function drive_gauss(t, wd, tg, A)
    if 0 <= t <= tg
         gauss_envelope(t, tg, A) * cos(wd * t)
    else
        0
    end
end