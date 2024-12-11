using Pkg
Pkg.activate(".")


# Will have to run once
ENV["PYTHON"] = "./anaconda3/bin/python"
# Pkg.build("PyCall")

using QuantumOptics, NPZ, PyPlot
include("gate_utils.jl")


# Truncate Hamiltonian Function
function truncate(H0, hspace)
    H0[hspace, hspace]
end


drive = "theta"
params = npzread("H_$drive.npz")

# The .+ 1 is elementwise addition to an array/vector
# Must do .+ 1 because Julia indexes from 1
hspace_full = params["hspace_full"] .+ 1
hspace_reduced = params["hspace_reduced"] .+ 1
w_trans_1 = params["w_trans_1"]
w_trans_2 = params["w_trans_2"]
drive_term = params["drive"]
H0 = params["H0"]

basis = NLevelBasis(length(hspace_reduced))
H_trunc = Operator(basis, truncate(H0, hspace_reduced))
drive_trunc = Operator(basis, truncate(drive_term, hspace_reduced))



H = LazySum([1.0, 0.0, 0.0], [H_trunc, drive_trunc, drive_trunc])
function Ht(t, psi)
    H.factors[2] = drive_gauss(t, params["drive_amp_A"], params["drive_freq_A"], params["gate_time"])
    H.factors[3] = drive_gauss(t, params["drive_amp_B"], params["drive_freq_B"], params["gate_time"])
    return H
end

# Expectation values
function calc_pops(t, psi)
    p1 = abs(psi.data[1])
    p2 =  abs(psi.data[2])
    p3 =  abs(psi.data[3])
    return p1, p2, p3
end

tlist = LinRange(0,params["gate_time"],100)
psi0 = nlevelstate(basis, 1)
tout, pops = timeevolution.schroedinger_dynamic(tlist, psi0, Ht; fout=calc_pops)

plt.clf()
for i in range(1,3)
    plt.plot(tlist, [x[i] for x in pops])
end
plt.savefig("theta_gate.png")





