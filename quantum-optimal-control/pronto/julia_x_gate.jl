using Pkg
Pkg.activate(".")

# Will have to run once
# ENV["PYTHON"] = "./anaconda3/bin/python"
# Pkg.build("PyCall")

using QuantumOptics, NPZ, PyPlot
include("gate_utils.jl")
# Truncate Hamiltonian Function
function truncate(H0, hspace)
    H0[hspace .+ 1, hspace .+ 1]
end



drive = "theta"
params = npzread("/home/eweissler/src/zp_data/H_$drive.npz")

# The .+ 1 is elementwise addition to an array/vector
# Must do .+ 1 because Julia indexes from 1
hspace_full = params["hspace_full"] 
hspace_reduced = params["hspace_reduced"][1:100]
# hspace_reduced = sort(0:40)
# for i in 1:10
#     append!(hspace_reduced, i)
# end
# append!(hspace_reduced, 4)
# append!(hspace_reduced, 38)
hspace_reduced = sort(unique(hspace_reduced))
println("hspace ", hspace_reduced)
# hspace_reduced = [0, 1, 2, 5, 7, 25, 38]
# hspace_reduced = hspace_full
w_trans_1 = params["w_trans_1"]
w_trans_2 = params["w_trans_2"]
drive_term = params["drive"]
H0 = params["H0"]

basis = NLevelBasis(length(hspace_reduced))
H_trunc = Operator(basis, truncate(H0, hspace_reduced))
drive_trunc = Operator(basis, truncate(drive_term, hspace_reduced))



H = LazySum([1.0, 0.0, 0.0], [H_trunc, drive_trunc, drive_trunc])
function Ht(t, psi)
    H.factors[2] = drive_gauss(t, params["drive_freq_A"], params["gate_time"], params["drive_amp_A"], normalized=false)
    H.factors[3] = drive_gauss(t, params["drive_freq_B"], params["gate_time"], params["drive_amp_B"], normalized=false)
    return H
end


tlist = LinRange(0,params["gate_time"], 500)

props, G, times = batch_evol(Ht, [0, 2], tlist)

U0 = zeros(ComplexF64, 2, 2)
U0[1,2] = 1
U0[2,1] = 1

fid = fid_coherent(U0, G)

println("fidelity ", fid)
println("log(1-fidelity) ", log10(1-fid))

pops, f = plot_evolution(hspace_reduced, tlist, props, [0, 2], 7, savename="theta_gate.png", s=3);

# plt.gca().set_yscale("log")
# plt.gca().set_ylim(1e-05, 1)
plt.savefig("theta_gate.png");

# #Expectation values
# function calc_pops(t, psi)
#     p1 = abs(psi.data[1])
#     p2 =  abs(psi.data[2])
#     p3 =  abs(psi.data[3])
#     return p1, p2, p3
# end

# psi0 = nlevelstate(basis, 1)
# tout, pops = timeevolution.schroedinger_dynamic(tlist, psi0, Ht; fout=calc_pops)

# f, ax = plt.subplots(ncols=2)
# ax[1].plot(tlist, drive_gauss.(tlist, params["drive_freq_A"], params["gate_time"], params["drive_amp_A"]))
# for i in range(1,3)
#     ax[2].plot(tlist, [x[i] for x in pops])
# end
# plt.draw()
# plt.savefig("theta_gate_2.png")
