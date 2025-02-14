using Pkg
Pkg.activate(".")

using QuantumOptics: Operator
using Statistics: mean

using PyPlot, NPZ

using PRONTO
using LinearAlgebra
using StaticArrays
using Base: @kwdef


include("gate_utils.jl")


# Plot results
function plot_results(res; savename = "control_plot_2.png", states_to_plot = [0, 2, 7], divide=10, fft = true)

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
    avg_val_1 = sum([mean(res_low_1), mean(res_high_1)])
    ax2.set_title("Max Other States $max_val_1" * " Avg $avg_val_1")

    ax3.plot(ts, res_low_2, linewidth = 2, label = "others < |$divide⟩")
    ax3.plot(ts, res_high_2, linewidth = 2, label = "others > |$divide⟩")
    max_val_2 = maximum([maximum(res_low_2), maximum(res_high_2)])
    avg_val_2 = sum([mean(res_low_2), mean(res_high_2)])
    ax3.set_title("Max Other States $max_val_2" * " Avg $avg_val_2")
    ax3.legend()

    plt.tight_layout()

    plt.savefig(savename)

end


## ----------------------------------- Imaginary/Real Helpers ----------------------------------- ##

function im_to_re(x::Union{Matrix{Num},Matrix{ComplexF64}})
    Re = I(2)
    Im = [0 -1;
          1 0]
    M = kron(Re,real(x)) + kron(Im,imag(x))
    return M
end

function im_to_re(x::Union{Vector{Num},Vector{ComplexF64}})
    return [real(x) ; imag(x)]
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

function quadratic_dist(x, xf)
    return 1/2 * (x-xf)' * I * (x-xf)
end

## ----------------------------------- PRONTO ----------------------------------- ##

function get_model_size(dim::Int, n_basis::Int)
    # Define size of the model
    # *2 for real, * number of basis states
    return 2*dim*n_basis
end

# model_size = 10
# n_drive = 1
# @kwdef struct model <: PRONTO.Model{model_size, n_drive}
# end

function gen_model!(model, H0::Matrix{ComplexF64}, H1::Vector{Matrix{ComplexF64}}, ψ0::Vector{Vector{ComplexF64}}, ψf::Vector{Vector{ComplexF64}}, penalty::Vector{Float64}, cost_function)

    # xf real, stacked
    xf = vcat([im_to_re(ψ) for ψ in ψf])

    # Define size of the model
    model_size = get_model_size(H0, H1, length(ψ0))

    eval("
    @define_f model begin
            H00 = im_to_re( -im *  H0)
            H11 = im_to_re.(-im .* H1)
            return kron(I(2), H00 + u .* H11) * x
    end
    
    @define_l model begin
        kl/2*u'*I*u + kq/2*x'*diagm([penalty; penalty])*x
    end
   
    @define_m model begin
        return cost_function(x, xf)
    end
    
    @define_Q model I(model_size)
    
    @define_R model I(1)
    ")
end



drive = "theta"
params = npzread("H_$drive.npz")

# The .+ 1 is elementwise addition to an array/vector
# Must do .+ 1 because Julia indexes from 1
hspace_full = params["hspace_full"] 
# hspace_reduced = params["hspace_reduced"] 
hspace_reduced = [0, 1, 2, 5, 7, 25, 38]
# hspace_reduced = [0, 1, 2, 7]
w_trans_1 = params["w_trans_1"]
w_trans_2 = params["w_trans_2"]
drive_term = params["drive"]
H0 = params["H0"]

hspace = hspace_reduced
basis = NLevelBasis(length(hspace))
H_trunc = Operator(basis, truncate(H0, hspace))
drive_trunc = Operator(basis, truncate(drive_term, hspace))

ψ1 = zeros(ComplexF64, length(hspace))
ψ1[findfirst(x->x==0, hspace)] = 1
ψ2 = zeros(ComplexF64, length(hspace))
ψ2[findfirst(x->x==2, hspace)] = 1

# To penalize all levels besides 0, 2, 7
qvec = ones(2*length(hspace))
for s in [0, 2, 7]
    qvec[findfirst(x->x==s, hspace)] = 0
    qvec[findfirst(x->x==s, hspace)+length(hspace)] = 0
end

H0 = dense(H_trunc).data
H1 = [dense(drive_trunc).data]
ψ0 = [ψ1, ψ2]
ψf = [ψ2, ψ1]
penalty = qvec
cost_function = quadratic_dist

# H0::Matrix{Complex}, H1::Vector{Matrix{Complex}}, ψ0::Vector{Matrix{Complex}}, ψf::Vector{Matrix{Complex}},
# penalty::Vector{Float}, cost_function::Function, kl::Float, kq::Float
model_size = get_model_size(size(H0)[1], length(ψ0))
n_drive = 1
@kwdef struct model <: PRONTO.Model{model_size, n_drive}
    kl::Float64 = 0.0005*2
    kq::Float64 = 0.006*2
end

# gen_model!(model, dense(H_trunc).data, [dense(drive_trunc).data], [ψ1, ψ2], [ψ2, ψ1], qvec, quadratic_dist)

@define_f model begin
    H = im_to_re( -im * H0)
    for i in eachindex(H1)
        H += u[i] * im_to_re( -im * H1[i])
    end
    return kron(I(2), H) * x
end

@define_l model begin
kl/2*u'*I*u + kq/2*x'*diagm([penalty; penalty])*x
end

# xf real, stacked
xf = SVector{model_size}(reduce(vcat, [im_to_re(ψ) for ψ in ψf]))
x0 = SVector{model_size}(reduce(vcat, [im_to_re(ψ) for ψ in ψ0]))
@define_m model begin
return cost_function(x, xf)
end

@define_Q model I(model_size)

@define_R model I(1)

PRONTO.Pf(θ::model,α,μ,tf) = SMatrix{model_size,model_size,Float64}(I(model_size))
resolve_model(model)

θ = model()


τ = t0,tf = 0,round(params["gate_time"], digits=3)

x0 = SVector{model_size}(x0)
μ = t->SVector{1}(drive_gauss(t, params["drive_freq_A"], params["gate_time"], params["drive_amp_A"], normalized=false) + drive_gauss(t, params["drive_freq_B"], params["gate_time"], params["drive_amp_B"], normalized=false))
η = open_loop(θ, x0, μ, τ) # guess trajectory




# plot_results(η; savename= "guess_trajectory.png", states_to_plot=[0, 1, 2, 5, 7, 25, 38])
plot_results(η; savename= "guess_trajectory.png", states_to_plot=[0, 2, 7])
ξ,data = pronto(θ, x0, η, τ;tol=1e-4); # optimal trajectory

## ----------------------------------- plot results ----------------------------------- ##


plot_results(ξ; savename= "optimal_trajectory.png")


res = ξ;
half = Int(length(res.x(τ[1]))/2);
res1_im = [re_to_im(res.x(t)[1:half]) for t in τ];
res2_im = [re_to_im(res.x(t)[half+1:end]) for t in τ];

println("0 pop in 2: ", abs.(res1_im[end][3]))
println("2 pop in 0: ", abs.(res2_im[end][1]))