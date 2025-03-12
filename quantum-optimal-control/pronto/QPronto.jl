using Pkg
Pkg.activate(".")

using Revise

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
    max_val_1 = round.([maximum(res_low_1), maximum(res_high_1)], digits=4)
    avg_val_1 = round.([mean(res_low_1), mean(res_high_1)], digits=4)
    ax2.set_title("Max Other States $max_val_1" * " Avg $avg_val_1")

    ax3.plot(ts, res_low_2, linewidth = 2, label = "others < |$divide⟩")
    ax3.plot(ts, res_high_2, linewidth = 2, label = "others > |$divide⟩")
    max_val_2 = round.([maximum(res_low_2), maximum(res_high_2)], digits=4)
    avg_val_2 = round.([mean(res_low_2), mean(res_high_2)], digits=4)
    ax3.set_title("Max Other States $max_val_2" * " Avg $avg_val_2")
    ax3.legend()

    plt.tight_layout()

    plt.savefig(savename)

end


## ----------------------------------- Imaginary/Real Helpers ----------------------------------- ##

function im_to_re(x::Matrix)
    Re = I(2)
    Im = [0 -1;
          1 0]
    M = kron(Re,real(x)) + kron(Im,imag(x))
    return M
end

function im_to_re(x::Vector)
    return [real(x) ; imag(x)]
end

function re_to_im(x::Union{Matrix, Matrix{Float64}})
    half = [Int(s/2) for s in size(x)]
    # Upper left quadrant real, lower left quantrant imaginary
    return x[1:half[1], 1:half[2]] + im*x[half[1]+1:end, 1:half[2]]
end

function re_to_im(x::Union{Vector, Vector{Float64}})
    half = Int(length(x)/2)
    # first half real, second half imaginary
    return x[1:half] + im*x[half+1:end]
end

function quadratic_dist(x, xf)
    return 1/2 * (x-xf)' * I * (x-xf)
end

function stack_x(xvec)
    return reduce(vcat, [im_to_re(x) for x in xvec])
end

function unstack_x(x, n)
    chunk = Int(length(x)/n)
    return [x[1+chunk*(i-1):chunk*i] for i in 1:n]
end

## ----------------------------------- PRONTO ----------------------------------- ##

function get_model_size(dim::Int, n_basis::Int)
    # Define size of the model
    # *2 for real, * number of basis states
    return 2*dim*n_basis
end

function gen_model(H0::Matrix{ComplexF64},
                   H1::Vector{Matrix{ComplexF64}},
                   ψ0::Vector{Vector{ComplexF64}},
                   ψf::Vector{Vector{ComplexF64}},
                   cost_function::Function,
                   level_penalty::Vector{Float64},
                   kl::Float64,
                   kq::Float64)

    # Define struct
    n_basis = length(ψ0)
    model_size = get_model_size(size(H0)[1], n_basis)
    n_drive = length(H1)
    eval(quote
        @kwdef struct model <: PRONTO.Model{$model_size, $n_drive}
                kl::Float64 = $kl #0.0005*2
                kq::Float64 = $kq #0.006*2
        end
    end
    )

    # Define necessary functions
    # Dynamics
    eval(quote
        @define_f model begin
            H = im_to_re( -im * $H0)
            for i in eachindex($H1)
                H += u[i] * im_to_re( -im * $H1[i])
            end
            return kron(I($n_basis), H) * x
        end
    end
    )
    ## TRYING IN INTERACTION PICTURE
#     eval(quote
#     @define_f model begin
#         H = u[1] * im_to_re( -im * $H1[1])
#         U = im_to_re(exp.(im * $H0 * t))
#         Udag = im_to_re(exp.(-im * $H0 * t))
#         for i in 2:length(u)
#             H .+= u[i] * im_to_re( -im * $H1[i])
#         end
#         return kron(I($n_basis), U*H*Udag) * x
#     end
# end
# )
    # Control + Level Penalties
    eval(quote
        @define_l model begin
            kl/2*u'*I*u + kq/2*x'*diagm(repeat($level_penalty, outer=2*$n_basis))*x
        end    
    end
    )
    # xf real, stacked
    # Cost function (state distance from desired final state)
    eval(quote
        @define_m model begin
            xf = SVector{$model_size}(stack_x($ψf))
            x0 = SVector{$model_size}(stack_x($ψ0))
            return $cost_function(x, xf)
        end
    end
    )


    eval(quote
        @define_Q model I($model_size)
    end
    )
    
    eval( quote
        @define_R model I(1)
    end
    )

    eval(:(PRONTO.Pf(θ::model,α,μ,tf) = SMatrix{$model_size,$model_size,Float64}(I($model_size))))
    eval(:(resolve_model(model)))

    return eval(:(model())) 
end