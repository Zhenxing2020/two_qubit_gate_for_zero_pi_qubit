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
# hspace_reduced = params["hspace_reduced"] 
# hspace_reduced = params["hspace_reduced"][1:15]
hspace_reduced = params["hspace_reduced"][1:19]
# append!(hspace_reduced, 4)
# append!(hspace_reduced, 38)
# hspace_reduced = [0, 1, 2, 5, 7, 25, 38]
# hspace_reduced = [0, 1, 2, 7]
hspace_reduced = sort(unique(hspace_reduced))
w_trans_1 = params["w_trans_1"]
w_trans_2 = params["w_trans_2"]
drive_term = params["drive"]
H0 = params["H0"]


# Different drives
# p = [20.041257,0.241329,0.059044,-0.019402,-0.009802]
# p = [30.008342,0.14538,0.037985,0.000555,0.004004]
# p = [39.991616,0.105045,0.028058,0.002171,0.003988]
# p = [50.273181,0.081831,0.022089,0.001944,0.003051]
# p = [59.087483,0.068953,0.018674,0.001617,0.002406]
p = [69.959457,0.05777,0.0157,0.001257,0.001811]

# Detuning -> drive frequency
p[4] = w_trans_1 + 2*pi*p[4]
p[5] = w_trans_2 + 2*pi*p[5]

fields = ["gate_time", "drive_amp_A", "drive_amp_B", "drive_freq_A", "drive_freq_B"]
for i in eachindex(fields)
    params[fields[i]] = p[i]
end

hspace = hspace_reduced
basis = NLevelBasis(length(hspace))
H_trunc = Operator(basis, truncate(H0, hspace))
drive_trunc = Operator(basis, truncate(drive_term, hspace))

ψ1 = zeros(ComplexF64, length(hspace))
ψ1[findfirst(x->x==0, hspace)] = 1
ψ2 = zeros(ComplexF64, length(hspace))
ψ2[findfirst(x->x==2, hspace)] = 1

# To penalize all levels besides 0, 2, 7
qvec = ones(length(hspace))
# for s in 1:9
#     if s in hspace
#         qvec[findfirst(x->x==s, hspace)] = 0.1
#     end
# end
for s in [0, 2]
    qvec[findfirst(x->x==s, hspace)] = 0
    # qvec[findfirst(x->x==s, hspace)+length(hspace)] = 0
end
for s in [7]
    qvec[findfirst(x->x==s, hspace)] = 0.1
    # qvec[findfirst(x->x==s, hspace)+length(hspace)] = 0.1*0
end
for s in [x for x in hspace if x > 9]
    qvec[findfirst(x->x==s, hspace)] = 2
end

H0 = dense(H_trunc).data
H1 = [dense(drive_trunc).data .+ 0.0im]
ψ0 = [ψ1, ψ2]
ψf = [ψ2, ψ1]
# ψ0 = [ψ1]
# ψf = [ψ2]
penalty = qvec

# X gate
U0 = zeros(ComplexF64, 2, 2)
U0[1,2] = 1
U0[2,1] = 1
function xgate_infid(x, xf)
    G = zeros(Number, 2, 2)
    x = unstack_x(x, 2)
    xf = reverse(unstack_x(xf, 2))
    for i in 1:2
    # for i in [1, 3]
        for j in 1:2
        # for j in [1, 2]
            val = re_to_im(xf[i])' * re_to_im(x[j])
            # println(val)
            G[i, j]  = val 
        # G[i, j] = re_to_im(x[j])[i]
        end
    end
    # return log10(1 - fid_coherent(U0, G)) 
    return 1 - fid_coherent(U0, G) 
    # return quadratic_dist(x, xf)
end

conj_op = kron([1 0; 0 -1], I(length(hspace)))
conj_op = kron(I(2), conj_op)
function quad_dist_phase(x, xf)
    # G = zeros(Number, 2, 2)
    x = unstack_x(x, 2)
    xf = unstack_x(xf, 2)
    phase = exp(-1.0im*angle(re_to_im(xf[1])'*re_to_im(x[1])))
    rval = 0
    for i in eachindex(xf)
        diff = re_to_im(xf[i]) - phase * re_to_im(x[i])
        rval += real((1/2)*diff'*diff)
        # rval += 1/2 - re_to_im(xf[i])
    end
    return rval
    # return re((xf-x)' * I * (xf-x))
    # conj_mat = 
    # 1/2 * (x-xf)' * I * (x-xf)


    
    # return log10(1 - fid_coherent(U0, G)) 
    # return 1 - fid_coherent(U0, G)
    # return quadratic_dist(x, xf)

    # return 1/2 * (x-xf)' * conj_op * (x-xf)
    # return 1/2 * (x-xf)' * I * (x-xf)
end



model_size = get_model_size(size(H0)[1], length(ψ0))
n_drive = 1

# kl = 0.0005*2#2
# kq = 0.006*2#2
kq = 0.006*2#2
# kq = 0.05
# kl = 0.01*10
kl = 0.01
# kq = 0.5

# gen_model(H0, H1, ψ0, ψf, xgate_infid, qvec, kl, kq)
# gen_model(H0, H1, ψ0, ψf, quadratic_dist, qvec, kl, kq)
gen_model(H0, H1,  ψ0, ψf, quad_dist_phase, qvec, kl, kq)


PRONTO.Pf(θ::model,α,μ,tf) = SMatrix{model_size,model_size,Float64}(I(model_size))
resolve_model(model)

θ = model()


τ = t0,tf = 0,round(params["gate_time"], digits=3)

xf = SVector{model_size}(stack_x(ψf))
x0 = SVector{model_size}(stack_x(ψ0))
μ = t->SVector{1}(drive_gauss(t, params["drive_freq_A"], params["gate_time"], params["drive_amp_A"], normalized=false) + drive_gauss(t, params["drive_freq_B"], params["gate_time"], params["drive_amp_B"], normalized=false))
η = open_loop(θ, x0, μ, τ) # guess trajectory




# plot_results(η; savename= "guess_trajectory.png", states_to_plot=[0, 1, 2, 5, 7, 25, 38])
plot_results(η; savename= "guess_trajectory_tg$tf.png", states_to_plot=[0, 2, 7])

ξ,data = pronto(θ, x0, η, τ;tol=1e-4, save_data=false); # optimal trajectory

## ----------------------------------- plot results ----------------------------------- ##


plot_results(ξ; savename= "optimal_trajectory_tg$tf.png")


res = ξ;
tf = ξ.u.itp.ranges[1][end]
half = Int(length(res.x(τ[1]))/2);
res1_im = [re_to_im(res.x(t)[1:half]) for t in τ];
res2_im = [re_to_im(res.x(t)[half+1:end]) for t in τ];

println("0 pop in 2: ", abs.(res1_im[end][3]))
println("2 pop in 0: ", abs.(res2_im[end][1]))

## ----------------------------------- QO.jl  ----------------------------------- ##

hr = [sort(hspace_reduced), sort(params["hspace_reduced"][1:100]), hspace_noise]

for i in eachindex(hr)
    hspace_reduced = hr[i]

    basis = NLevelBasis(length(hspace_reduced))
    H_trunc = Operator(basis, truncate(params["H0"], hspace_reduced))
    drive_trunc = Operator(basis, truncate(params["drive"], hspace_reduced))

    H = LazySum([1.0, 0.0], [H_trunc, drive_trunc])
    function Ht(t, psi)
        H.factors[2] = res.u(t)[1]
        # H.factors[2] = η.u(t)[1]
        return H
    end


    tlist = LinRange(0,tf, 500)

    props, G, times = batch_evol(Ht, [0, 2], tlist)

    U0 = zeros(ComplexF64, 2, 2)
    U0[1,2] = 1
    U0[2,1] = 1

    fid = fid_coherent(U0, G)
    println("$i -------------------")
    println("fidelity ", fid_coherent(U0, G))
    println("fidelity (no tr) ", fid_coherent2(U0, G))
    println("log(1-fidelity) ", log10(1-fid))
    println("log(1-fidelity no tr) ", log10(1-fid_coherent2(U0, G)))

    pops, f = plot_evolution(hspace_reduced, tlist, props, [0, 2], 7, savename="optimal_trajectory_qo_$i.png", s=1);

end

fname = "solution_tg$tf,kq$kq,kl$kl.bson"

BSON.@save fname ξ
BSON.@load fname ξ