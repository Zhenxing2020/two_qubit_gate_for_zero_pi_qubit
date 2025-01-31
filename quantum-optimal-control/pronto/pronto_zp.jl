using Pkg
Pkg.activate(".")

include("gate_utils.jl")

using PyPlot, NPZ

using PRONTO
using LinearAlgebra
using StaticArrays
using Base: @kwdef

## ----------------------------------- define helper functions ----------------------------------- ##

function mprod(x)
    Re = I(2)
    Im = [0 -1;
          1 0]
    M = kron(Re,real(x)) + kron(Im,imag(x))
    return M
end

function re_to_im(x::Union{Matrix{Num},Matrix{Float64}})
    half = [Int(s/2) for s in size(x)]
    # Upper left quadrant real, lower left quantrant imaginary
    return x[1:half[1], 1:half[2]] + im*x[half[1]+1:end, 1:half[2]]
end

function re_to_im(x::Union{Vector{Num},Vector{Float64}})
    half = Int(length(x)/2)
    # first half real, second half imaginary
    return x[1:half] + im*x[half+1:end]
end

## ----------------------------------- Load data from python ----------------------------------- ##

drive = "theta"
params = npzread("H_$drive.npz")

# The .+ 1 is elementwise addition to an array/vector
# Must do .+ 1 because Julia indexes from 1
hspace_full = params["hspace_full"] 
# hspace_reduced = params["hspace_reduced"] 
hspace_reduced = [0, 1, 2, 5, 7, 25, 38]
w_trans_1 = params["w_trans_1"]
w_trans_2 = params["w_trans_2"]
drive_term = params["drive"]
H0 = params["H0"]


function truncate(H0, hspace)
    H0[hspace .+ 1, hspace .+ 1]
end

basis = NLevelBasis(length(hspace_reduced))
H_trunc = Operator(basis, truncate(H0, hspace_reduced))
drive_trunc = Operator(basis, truncate(drive_term, hspace_reduced))

hspace = hspace_reduced
H = H_trunc
drive = drive_trunc


# Plot results
function plot_results(res; savename = "control_plot_2.png", states_to_plot = [0, 2, 7], divide=10)

    fig, ax = plt.subplots(nrows=3, figsize=(10, 10))
    t0,tf = τ
    ts = range(t0,tf,length=1001)

    (ax1, ax2, ax3) = ax

    for a in ax
        a.set_xlabel("time")
    end

    ax1.set_ylabel("control input")
    ax2.set_ylabel("population")
    ax3.set_ylabel("population")

    n = length(hspace)

    ax1.plot(ts, [res.u(t)[1] for t in ts], linewidth = 2)

    others_low = []
    others_high = []
    for s in hspace
        if !(s in states_to_plot)
            if s < divide
                push!(others_low, s)
            else
                push!(others_high, s)
            end
        end
    end

    half = Int(length(res.x(ts[1]))/2)
    res1_im = [re_to_im(res.x(t)[1:half]) for t in ts]
    res2_im = [re_to_im(res.x(t)[half+1:end]) for t in ts]

    for s in states_to_plot
        i = findfirst(x->x==s, hspace)
        ax2.plot(ts, [abs(x[i])^2 for x in res1_im], linewidth = 2, label = "|$s⟩")
        ax3.plot(ts, [abs(x[i])^2 for x in res2_im], linewidth = 2, label = "|$s⟩")
    end
    
    i = 1
    res_low_1 = 0*[res.x(t)[2*n+i]^2+res.x(t)[2*n+i+n]^2 for t in ts]
    res_high_1 = 0*[res.x(t)[2*n+i]^2+res.x(t)[2*n+i+n]^2 for t in ts]
    res_low_2 = 0*[res.x(t)[2*n+i]^2+res.x(t)[2*n+i+n]^2 for t in ts]
    res_high_2 = 0*[res.x(t)[2*n+i]^2+res.x(t)[2*n+i+n]^2 for t in ts]

    for s in others_low
        i = findfirst(x->x==s, hspace)
        res_low_1  = res_low_1 + [abs(x[i])^2 for x in res1_im]
        res_low_2  = res_low_2 + [abs(x[i])^2 for x in res2_im]
    end
    for s in others_high
        i = findfirst(x->x==s, hspace)
        res_high_1 = res_high_1 +  [abs(x[i])^2 for x in res1_im]
        res_high_2 = res_high_2 +  [abs(x[i])^2 for x in res2_im]
    end
    ax2.plot(ts, res_low_1, linewidth = 2, label = "others < |$divide⟩")
    ax2.plot(ts, res_high_1, linewidth = 2, label = "others > |$divide⟩")
    ax2.legend()
    max_val_1 = maximum([maximum(res_low_1), maximum(res_high_1)])
    ax2.set_title("Max Other States $max_val_1")

    ax3.plot(ts, res_low_2, linewidth = 2, label = "others < |$divide⟩")
    ax3.plot(ts, res_high_2, linewidth = 2, label = "others > |$divide⟩")
    max_val_2 = maximum([maximum(res_low_2), maximum(res_high_2)])
    ax3.set_title("Max Other States $max_val_2")
    ax3.legend()

    plt.tight_layout()

    plt.savefig(savename)

end



## ----------------------------------- define the model ----------------------------------- ##

# kq for level penalization
# kl for control effort penalization
n_basis = 2
model_size = (length(hspace))*2*n_basis
@kwdef struct XGateZP <: PRONTO.Model{model_size,1}
    kl::Float64 = 0.01
    kq::Float64 = 0.5
    # kq::Float64 = 0.5/4.5
    # Estimates based off fidelity
    # kl::Float64 = 0.0005*2
    # kq::Float64 = 0.006*2
end


# The kron(I(2)) is to have it act on two states
# at the same time
@define_f XGateZP begin
    H00 = mprod(-im * H.data)
    H11 = mprod(-im * drive.data)
    return kron(I(2), H00 + u[1]*H11) * x
    # return 2 * π * mprod(-im * (H00 + u[1]*H11)) * x
    # return (1/(2 * π)) * mprod(-im * (H00 + u[1]*H11)) * x
    # return 2 * π * mprod(-im * (H00 + H11)) * x
end

# To penalize all levels besides 0, 2, 7
qvec = ones(2*length(hspace))
for s in [0, 2, 7]
    qvec[findfirst(x->x==s, hspace)] = 0
    qvec[findfirst(x->x==s, hspace)+length(hspace)] = 0
end
@define_l XGateZP begin
    kl/2*u'*I*u + kq/2*x'*diagm([qvec;qvec])*x
end

@define_m XGateZP begin
    ψ1 = zeros(length(hspace))
    i0 = findfirst(i->i==0, hspace)
    ψ1[i0] = 1
    ψ2 = zeros(length(hspace))
    i2 = findfirst(i->i==2, hspace)
    ψ2[i2] = 1
    xf = vec([ψ2;0*ψ2;ψ1;0*ψ1])

    # Turn x into an imaginary vector for each state
    half = Int(size(x)[1]/2)
    # x1 = abs.(re_to_im(x[1:half]))
    # x2 = abs.(re_to_im(x[half+1:end]))
    x1 = re_to_im(x[1:half])
    x2 = re_to_im(x[half+1:end])

    phase = exp(-im*angle(ψ2'*x1))

    idx = [i0, i2]
    G = phase*hcat(x1[idx], x2[idx])
    sx = hcat([0.0, 1.0], [1.0, 0.0])
    d = 2
    fid = 1/(d+1) + (1/(d*(d+1)))*abs(tr(sx*G))^2


    return (1/2)*(1 - abs(ψ1'*x2)^2 - abs(ψ2'*x1)^2)



    # return (1/2)*((ψ2-x1)'*(ψ2-x1) + (ψ1-x2)'*(ψ1-x2))
    # print("\n new\n")
    # print((1/2)*((ψ2-x1)'*(ψ2-x1) + (ψ1-x2)'*(ψ1-x2)))
    # print("\n old\n")
    # print(1/2*(x-xf)'*I(model_size)*(x-xf))
    # return 1/2*(x-xf)'*I(model_size)*(x-xf)
end


@define_Q XGateZP I(model_size)
@define_R XGateZP I(1)
PRONTO.Pf(θ::XGateZP,α,μ,tf) = SMatrix{model_size,model_size,Float64}(I(model_size))

resolve_model(XGateZP)

## ----------------------------------- run optimization ----------------------------------- ##

θ = XGateZP()
τ = t0,tf = 0,round(params["gate_time"], digits=3)
ψ1 = zeros(length(hspace))
ψ1[findfirst(x->x==0, hspace)] = 1
ψ2 = zeros(length(hspace))
ψ2[findfirst(x->x==2, hspace)] = 1
x0 = SVector{model_size}(vec([ψ1;0*ψ1;ψ2;0*ψ2]))
μ = t->SVector{1}(drive_gauss(t, params["drive_freq_A"], params["gate_time"], params["drive_amp_A"]) + drive_gauss(t, params["drive_freq_B"], params["gate_time"], params["drive_amp_B"]))
η = open_loop(θ, x0, μ, τ) # guess trajectory


plot_results(η; savename= "guess_trajectory.png", states_to_plot=[0, 1, 2, 5, 7, 25, 38])
ξ,data = pronto(θ, x0, η, τ;tol=1e-4); # optimal trajectory

## ----------------------------------- plot results ----------------------------------- ##


plot_results(ξ; savename= "optimal_trajectory.png")


res = ξ;
half = Int(length(res.x(τ[1]))/2);
res1_im = [re_to_im(res.x(t)[1:half]) for t in τ];
res2_im = [re_to_im(res.x(t)[half+1:end]) for t in τ];

println("0 pop in 2: ", abs.(res1_im[end][3]))
println("2 pop in 0: ", abs.(res2_im[end][1]))


