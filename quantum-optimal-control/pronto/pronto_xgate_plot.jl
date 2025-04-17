using Pkg
Pkg.activate(".")

using Revise
using PRONTO

using BSON


include("gate_utils.jl")
include("QPronto.jl")

drive = "theta"
params = npzread("/home/eweissler/src/zp_data/H_$drive.npz")


# The .+ 1 is elementwise addition to an array/vector
# Must do .+ 1 because Julia indexes from 1
hspace_full = params["hspace_full"] 
hspace_noise = [
    0, 1, 2, 4, 5, 7, 8, 11, 12, 16 ,
    17, 18, 21, 23, 25, 27, 30, 31, 32, 33 ,
    37, 38, 40, 41, 42, 45, 46, 47, 48, 51 ,
    52, 56, 59, 62, 63, 64, 65, 67, 72, 73 ,
    74, 76, 78, 79, 82, 83, 85, 87, 88, 90 ,
    92, 93, 96, 97, 98, 102, 103, 106, 107, 112 ,
    114, 118, 119, 120, 121, 122, 123, 124, 127, 128 ,
    131, 132, 133, 134, 137, 138, 142, 143 ,
    ]
hspace_reduced = hspace_noise
# hspace_reduced = params["hspace_reduced"] 
# hspace_reduced = params["hspace_reduced"][1:15]
# hspace_reduced = params["hspace_reduced"][1:19]
# append!(hspace_reduced, 4)
# append!(hspace_reduced, 38)
# hspace_reduced = [0, 1, 2, 5, 7, 25, 38]
# hspace_reduced = [0, 1, 2, 7]
hspace_reduced = sort(unique(hspace_reduced))
w_trans_1 = params["w_trans_1"]
w_trans_2 = params["w_trans_2"]
drive_term = params["drive"]
H0 = params["H0"]





##################### DEFINE MODEL FOR BSON
hspace = params["hspace_reduced"][1:3]
basis = NLevelBasis(length(hspace))
H_trunc = Operator(basis, truncate(H0, hspace))
drive_trunc = Operator(basis, truncate(drive_term, hspace))
ψ1 = zeros(ComplexF64, length(hspace))
ψ1[findfirst(x->x==0, hspace)] = 1
ψ2 = zeros(ComplexF64, length(hspace))
ψ2[findfirst(x->x==2, hspace)] = 1
qvec = ones(length(hspace))
for s in [0, 2]
    qvec[findfirst(x->x==s, hspace)] = 0
end
for s in [7]
    qvec[findfirst(x->x==s, hspace)] = 0.1
end
for s in [x for x in hspace if x > 9]
    qvec[findfirst(x->x==s, hspace)] = 2
end
H0 = dense(H_trunc).data
H1 = [dense(drive_trunc).data .+ 0.0im]
ψ0 = [ψ1, ψ2]
ψf = [ψ2, ψ1]
penalty = qvec
U0 = zeros(ComplexF64, 2, 2)
U0[1,2] = 1
U0[2,1] = 1
# function xgate_infid(x, xf)
#     G = zeros(Number, 2, 2)
#     x = unstack_x(x, 2)
#     xf = reverse(unstack_x(xf, 2))
#     for i in 1:2
#         for j in 1:2
#             val = re_to_im(xf[i])' * re_to_im(x[j])
#             G[i, j]  = val 
#         end
#     end
#     return 1 - fid_coherent(U0, G) 
# end

# conj_op = kron([1 0; 0 -1], I(length(hspace)))
# conj_op = kron(I(2), conj_op)
function quad_dist_phase(x, xf)
    x = unstack_x(x, 2)
    xf = unstack_x(xf, 2)
    phase = exp(-1.0im*angle(re_to_im(xf[1])'*re_to_im(x[1])))
    rval = 0
    for i in eachindex(xf)
        diff = re_to_im(xf[i]) - phase * re_to_im(x[i])
        rval += real((1/2)*diff'*diff)
    end
    return rval
end
model_size = get_model_size(size(H0)[1], length(ψ0))
n_drive = 1
kq = 0.006*2#2
kl = 0.01
gen_model(H0, H1,  ψ0, ψf, quad_dist_phase, qvec, kl, kq)



# Different drives
drives = Dict() 
drives["20"] = [20.041257,0.241329,0.059044,-0.019402,-0.009802]
drives["30"] = [30.008342,0.14538,0.037985,0.000555,0.004004]
drives["40"] = [39.991616,0.105045,0.028058,0.002171,0.003988]
drives["50"] = [50.273181,0.081831,0.022089,0.001944,0.003051]
drives["60"] = [59.087483,0.068953,0.018674,0.001617,0.002406]
drives["70"] = [69.959457,0.05777,0.0157,0.001257,0.001811]

# Detuning -> drive frequency
for t in keys(drives)
    p = drives[t]
    p[4] = w_trans_1 + 2*pi*p[4]
    p[5] = w_trans_2 + 2*pi*p[5]
    drives[t] = p
end

fields = ["gate_time", "drive_amp_A", "drive_amp_B", "drive_freq_A", "drive_freq_B"]
for t in keys(drives)
    pdict = Dict()
    p = drives[t]
    for i in eachindex(fields)
        pdict[fields[i]] = p[i]
    end
    drives[t] = pdict
end

fnames = Dict()
fnames["20"] = "solution_tg20.041,kq0.012,kl0.01.bson"
fnames["30"] = "solution_tg30.008,kq0.012,kl0.01.bson"
fnames["40"] = "solution_tg39.992,kq0.012,kl0.01.bson"
fnames["50"] = "solution_tg50.273,kq0.012,kl0.01.bson"
fnames["60"] = "solution_tg59.087,kq0.012,kl0.01.bson"
fnames["70"] = "solution_tg69.959,kq0.012,kl0.01.bson"


# Each pulse
pops_pronto = Dict{String, Array}()
pops_gauss = Dict{String, Array}()
# For popluations indices are logical_index, 0/2/7/25/other, time
pops_pronto["hspace"] = hspace_reduced
pops_gauss["hspace"] = hspace_reduced
basis = NLevelBasis(length(hspace_reduced))
H_trunc = Operator(basis, truncate(params["H0"], hspace_reduced))
drive_trunc = Operator(basis, truncate(params["drive"], hspace_reduced))
H = LazySum([1.0, 0.0], [H_trunc, drive_trunc])
# to_plot = [0, 2, 7, 25]
to_plot = hspace_reduced
sigma = 0
plt.close("all")
for t in keys(drives)

    BSON.@load fnames[t] ξ
    function Ht(t, psi)
        H.factors[2] = ξ.u(t)[1]
        return H
    end
    tf = ξ.u.itp.ranges[1][end]
    tlist = LinRange(0,tf, 500)
    props, G, times = batch_evol(Ht, [0, 2], tlist)
    population, fig = plot_evolution2(hspace_reduced, tlist, props, [0, 2], to_plot, savename="optimal_trajectory_qo_$t.png", s=sigma);
    pops_pronto[t] = population
    pops_pronto["tlist_$t"] = tlist


    dp = drives[t]
    function Ht_gauss(t, psi)
        H.factors[2] = drive_gauss(t, dp["drive_freq_A"], dp["gate_time"], dp["drive_amp_A"], normalized=false) + drive_gauss(t, dp["drive_freq_B"], dp["gate_time"], dp["drive_amp_B"], normalized=false)
        return H
    end
    tf = dp["gate_time"]
    tlist = LinRange(0,tf, 500)
    props, G, times = batch_evol(Ht_gauss, [0, 2], tlist)
    population, fig = plot_evolution2(hspace_reduced, tlist, props, [0, 2], to_plot, savename="optimal_trajectory_qo_$t-g.png", s=sigma);
    pops_gauss[t] = population
    pops_gauss["tlist_$t"] = tlist
end

npzwrite("pop_gauss$suffix.npz", pops_gauss)
npzwrite("pop_pronto$suffix.npz", pops_pronto)

# Plot populations
times = sort([x for x in keys(drives)])
# indices are pronto/gauss, 7/25/other, time
pops_avg = zeros(2, 2, length(times))
pops_max = zeros(2, 2, length(times))
for (it, t) in enumerate(times)
    offset = 3
    for j in [4, 5]
        pops_avg[1, j-offset, it] = mean(pops_gauss[t][:, j, :])
        pops_max[1, j-offset, it] = maximum(pops_gauss[t][:, j, :])
        pops_avg[2, j-offset, it] = mean(pops_pronto[t][:, j, :])
        pops_max[2, j-offset, it] = maximum(pops_pronto[t][:, j, :])
    end
end
plt.close("all")
f, ax = plt.subplots(ncols=2, nrows=2)
state_labels = ["25", "other"]
drive_labels = ["gauss", "pronto"]
for (il, l) in enumerate(state_labels)
    for (id, d) in enumerate(drive_labels)
        ax[il, 1].plot(times, pops_avg[id, il, :], label="$d")
        ax[il, 2].plot(times, pops_max[id, il, :], label="$d")
    end
    ax[il, 1].set_title("Avg in $l")
    ax[il, 2].set_title("Max in $l")
    for j in [1, 2]
        ax[il, j].legend()
        # ax[il, j].set_yscale("log")
        if il == 2
            ax[il, j].set_xlabel("gate time (ns)")
        end
        if j == 1
            ax[il, j].set_ylabel("population")
        end
    end
end
ax[1,1].set_ylim((0, 0.04))
ax[1,2].set_ylim((0, 0.11))
ax[2,1].set_ylim((0, 0.1))
ax[2,2].set_ylim((0, 0.5))
plt.tight_layout()

plt.savefig("leakage_pronto.png")

suffix = "_no_smooth"
npzwrite("pop$suffix.npz", Dict("pops_max" => pops_max, "pops_avg" => pops_avg))