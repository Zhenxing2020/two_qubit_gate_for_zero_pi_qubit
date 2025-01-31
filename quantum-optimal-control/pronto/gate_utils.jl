using QuantumOptics
using PyPlot



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

function batch_evol(Ht, logical_states, tlist)

    # Initial vectors and results/times to save
    psi0 = [nlevelstate(Ht(0, 0).basis_r, logical_states[i] + 1) for i in 1:length(logical_states)]
    res = Vector{Any}(undef, length(logical_states))
    times = Vector{Any}(undef, length(logical_states))

    # Do schrodinger equation evolution
    # Threads.@threads for i = 1:length(logical_states)
    for i = 1:length(logical_states)
        stuff = timeevolution.schroedinger_dynamic(tlist, psi0[i], Ht)
        times[i] = stuff[1]
        res[i] = stuff[2]
    end
    
    res, times
end


function plot_evolution(hspace, tlist, props, logical, intermediate; divide=11, suptitle="", savename="")
    
    # Get population of logical states, intermediate states, and other states < or >= divide
    pops = zeros((2, 5, length(tlist)))
    logical_idx = [findfirst(x->x==i, hspace) for i in logical]
    int_idx = findfirst(x->x==intermediate, hspace)
    dim = props[1][1].data.size[1]
    others_low = []
    others_high = []
    for i = 1:dim
        if (i in logical_idx) | (i == int_idx)
            continue
        end
        if i < divide
            push!(others_low, i)
        else
            push!(others_high, i)
        end

    end
    print(logical_idx, " ", int_idx, " ", others_low, " ", others_high)
    f, ax = plt.subplots(ncols=2, figsize=(8,3))
    for i = range(1, length(logical_idx))
        props_i = props[i]
        pops[i,1,:] = abs.([p.data[logical_idx[1]] for p in props_i]).^2
        pops[i,2,:] = abs.([p.data[logical_idx[2]] for p in props_i]).^2
        pops[i,3,:] = abs.([p.data[int_idx] for p in props_i]).^2
        for o in others_low
            pops[i,4,:] += abs.([p.data[o] for p in props_i]).^2
        end
        for o in others_high
            pops[i,5,:] += abs.([p.data[o] for p in props_i]).^2
        end

        ax[i].plot(tlist, pops[i, 1, :], label=logical[1])
        ax[i].plot(tlist, pops[i, 2, :],label=logical[2])
        ax[i].plot(tlist, pops[i, 3, :], label=intermediate)
        ax[i].plot(tlist, pops[i, 4, :], label="others < $divide")
        ax[i].plot(tlist, pops[i, 5, :], label="others >= $divide")
        ax[i].legend()
        start = hspace[i]
        ax[i].set_title("Start in $start")
        ax[i].set_xlabel("time (ns)")
        ax[i].set_ylabel("population")
    end
    if suptitle != ""
        plt.suptitle(suptitle)
    end
    if savename != ""
        plt.savefig(savename)
    end
    return pops, f
end