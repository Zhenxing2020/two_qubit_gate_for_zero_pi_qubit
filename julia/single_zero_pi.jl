### A Pluto.jl notebook ###
# v0.20.3

using Markdown
using InteractiveUtils

# ╔═╡ e195097a-f66f-11ed-3515-5bdb3424945b
begin
	import Pkg
	Pkg.activate(".")
	Pkg.add("PyPlot")
	Pkg.add("QuantumOptics")
	Pkg.add("Arpack")
	Pkg.add("LaTeXStrings")
	Pkg.add("PlutoUI")
	Pkg.add("LazyGrids")
end

# ╔═╡ d1559695-caab-4f27-a9d6-f0fb8eda7adf
begin
	using PyPlot
	using QuantumOptics
	using LinearAlgebra
	using LaTeXStrings
	using PlutoUI
	using LazyGrids: ndgrid_array
	import PlutoUI: combine
	include("cqed.jl")
end

# ╔═╡ a493192e-9026-4cc5-a270-9a7c092ef1de
md"""

#### Model the 0-pi Hamiltonian:


$$\begin{align}
&\hat{H}_{0-\pi} = 4 E_C^\theta(\hat{n}_\theta - n_g^\theta) + 4E_C^\phi \hat{n}^2_\phi - 2E_J\cos(\hat{\theta})\cos(\hat{\phi}-\frac{\pi \Phi_{ext}}{\Phi_0}) + E_L\hat{\phi}^2 
\end{align}$$

with drive of:

$$\begin{align}
\hat{H}_{drive} \approx (\beta_\phi \hat{n}_\phi + \beta_\theta \hat{n}_\theta)
\end{align}$$

Currently not considering the asymetric juntion term:

$$\begin{align}
+ E_J dE_J \sin(\hat{\theta})\sin(\hat{\phi}-\frac{\pi \Phi_{ext}}{\Phi_0})
\end{align}$$

or the asymetric capacitance term

$$\begin{align}
+ \hbar g_{\phi \theta} \hat{n}_\phi \hat{n}_\theta
\end{align}$$

"""

# ╔═╡ 6ab4542b-da76-4ac9-8408-a319f4aebc7d
begin
	# Parameters from https://journals.aps.org/prxquantum/pdf/10.1103/PRXQuantum.2.010339
	# Units of GHz*h
	c = 2*pi
	Ec_theta = 0.092*c
	Ec_phi = 1.142*c
	Ej = 6.013*c
	dEj = 0.1*c
	El = 0.377*c
	beta_phi = 0.27*c
	beta_theta = c*6.6*10^-3
	ng_theta = 0.25
	phi_ext = 0.0
	phi_range = [-4*pi, 4*pi]
	theta_range = [-pi/2, 3*pi/2]
	n_points = 50

	# Define bases
	phi_basis = PositionBasis(phi_range[1], phi_range[2], n_points)
	phi_points = samplepoints(phi_basis)
	theta_basis = PositionBasis(theta_range[1], theta_range[2], n_points)
	theta_points = samplepoints(theta_basis)

	

	# for plotting
	Y,X = ndgrid_array(theta_points,phi_points)

	# Make the ZP Hamiltonian
	Hzp = H_zero_pi(Ej, dEj, Ec_theta, Ec_phi, El, ng_theta, phi_ext, theta_basis, phi_basis)
	V = reshape(diag(real.((Hzp.operators[3]+Hzp.operators[4]).data)),n_points,n_points)
	Hzp = sparse(Hzp)
	
	# Define some operators
	n_theta = momentum(theta_basis)
	id_theta = identityoperator(n_theta)
	n_phi = momentum(phi_basis)
	id_phi = identityoperator(n_phi)
	
	# Make the drive Hamiltonian
	Hdrive = sparse(beta_phi*tensor(id_theta,n_phi) + beta_theta*tensor(n_theta, id_phi))
end

# ╔═╡ e219c5b2-7c19-41e7-96bf-f0099c5bbfb5
begin
	ev, es = eigenstates(Hzp, 50)
	shift = ev[1]
	ev = real.(ev .- shift)
end

# ╔═╡ 045b30e1-e9ef-4df7-8d8a-20c979fecb1a
md"""
#### Plot the potential energy and eigenstates:

The computational states $|0\rangle_L = |0\rangle$ and $|1\rangle_L = |2\rangle$ are spatially disjoint. $|0\rangle$ is localized in a single well at $\theta = 0$, while $|2\rangle$ is a symmetric combination of the potential wells at $\theta = \pi$.
"""

# ╔═╡ a488bc91-d674-44e5-a544-7a747ca1fc01
begin
	plt.close("all")
	surf = plot_surface(X,Y,V, cmap="plasma")
	plt.xlim((-3*pi,3*pi))
	plt.xlabel(L"\phi",fontsize=20)
	plt.ylabel(L"\theta",fontsize=20)
	plt.gca().set_zlabel(L"V(\theta, \phi)",fontsize=16)
	plt.colorbar(surf,shrink=0.4)
	# plt.savefig("zp_potential.svg")
	plt.gcf().set_size_inches(10,10)
	plt.gca().tick_params(axis="both", labelsize=14)
	# plt.tight_layout()
	plt.gcf()
end

# ╔═╡ c6e96ae2-df23-4a58-9c93-46e5720d8c3a
begin

	# Values to fix the colormap across eigenstates
	minval = minimum([minimum(real(x.data)) for x in es[1:9]])
	maxval = maximum([minimum(real(x.data)) for x in es[1:9]])
	spread = maximum([-minval, maxval])

	n_to_plot = 21
	n_per_col = 3
	n_rows = Integer(n_to_plot/n_per_col)
	f, ax = plt.subplots(nrows=n_rows,ncols=n_per_col)
	f.set_size_inches(n_per_col*3,n_rows*4/3)
	
	for i in range(1,n_rows)
		for j in range(1, n_per_col)
			ii = (i-1)*n_per_col + j
			toplot = real.(es[ii].data)
			# ax[i,j].pcolormesh(X,Y, reshape(toplot,(n_points, n_points)),cmap="bwr",vmin=-spread,vmax=spread)
			minval = minimum(toplot)
			maxval = maximum(toplot)
			spread = maximum([-minval, maxval])
			ax[i,j].pcolormesh(X,Y, reshape(toplot,(n_points, n_points)),cmap="bwr",vmin=-spread,vmax=spread)
			ax[i,j].set_xlabel(L"\phi")
			ax[i,j].set_ylabel(L"\theta")
			ax[i,j].set_title(L"$|"*string(ii-1)*L"\rangle$"* "  " * string(round(ev[ii]/(2*pi),digits=2))*L" $h\cdot$GHz")
			ax[i,j].set_aspect(1)
		end
	end
	plt.tight_layout()
	plt.gcf()
	
end

# ╔═╡ 5f7f7b2f-cf2a-489a-bc2b-1561b6d0e0e6
md"""
#### Three level dynamics:
"""

# ╔═╡ 48f125e4-7c86-49bd-9fc2-e1262b86abe8
begin
	function plot_mat_elems(mat, to_show, title = L"$\log_{10}H_{drive}$")
		plt.close("all")
		to_plot = mat_elems(es[to_show], Hdrive)
		plt.imshow(log10.(abs.(to_plot)))
		plt.title(title)
		plt.colorbar()
		kets = [L"$|"*string(x-1)*L"\rangle$" for x in to_show]
		plt.gca().set_xticks(range(0,length(to_show)-1))
		plt.gca().set_xticklabels(kets)
		plt.gca().set_yticks(range(0,length(to_show)-1))
		plt.gca().set_yticklabels(kets)
		plt.gcf()
	end
	plot_mat_elems(Hdrive, [1,3,10])
end

# ╔═╡ 7d8c18e2-7028-4de9-a0e6-e98715f3b78e
md"""
#### Determine some extra states to add to the three level dynamics:
"""

# ╔═╡ 5bc6e7b8-0fd8-40f2-b7d4-0603edcc3c62
begin
	all_elems = abs.(mat_elems(es, Hdrive))
	states_to_plot = [1,3,10]
	imax = zeros(size(states_to_plot)[1])
	plt.close("all")
	for (i, state) in enumerate(states_to_plot)
		y = abs.(all_elems[state,:])
		plt.plot(y, label = L"$|"*string(state-1)*L"\rangle$",alpha=0.8)
		imax[i] = findmax(y)[2]
	end
	plt.xlabel(L"State Number $n$")
	plt.ylabel(L"$|\langle n |H_{drive}| i \rangle|$ for specified state $i$")
	plt.legend()
	plt.gcf()
end

# ╔═╡ e094238f-aa9c-48e7-9235-d9dbd5e3b8f6
imax

# ╔═╡ 18c0cf62-fd9e-46f8-9fc9-a3aedde69f0c
md"""
The $|0\rangle$ and $|1\rangle$ states each have a large matrix element with a higher state. Try adding in one or both of them:
"""

# ╔═╡ 4a3d3de2-64d5-4f91-b8f5-29b47e5527fa
begin
	plt.close("all")
	to_show = [1,3,10,12,19]
	# to_show = [1,3,10]
	to_plot = mat_elems(es[to_show], Hdrive)
	plt.imshow(log10.(abs.(to_plot)))
	# plt.imshow(abs.(to_plot))
	plt.title(L"$\log_{10}H_{drive}$")
	plt.colorbar()
	kets = [L"$|"*string(x-1)*L"\rangle$" for x in to_show]
	plt.gca().set_xticks(range(0,length(to_show)-1))
	plt.gca().set_xticklabels(kets)
	plt.gca().set_yticks(range(0,length(to_show)-1))
	plt.gca().set_yticklabels(kets)
	plt.gcf()
end

# ╔═╡ d521009d-53bf-4430-9ab2-1f45e6c2eae8
round.(abs.(mat_elems(es[to_show], Hdrive)), digits=3)

# ╔═╡ b645f16b-5588-43c6-a7ed-aa92cf370f00
md"""
#### Do a simple two tone gate to transition from state $|0\rangle \rightarrow |1\rangle$:

First truncate the operators into the 5 level subspace (and set the zero energy of the Hamiltonian to the ground state):
"""

# ╔═╡ 877fb0cf-e3aa-4c89-94f8-d664a6d3a57c
begin
	proj, trunc_basis = trunc_subspace(es[to_show])
	const H_full = LazySum([1.0, 0.0], hermitify.([trunc_op(proj, Hzp)-identityoperator(trunc_op(proj, Hzp))*shift, trunc_op(proj, Hdrive)]))
end

# ╔═╡ 2e94f431-5451-488d-ab8e-71a2f3e45b45
md"""
Now create a two tone drive of the form:

$$\begin{align}
g(t)(\Omega_a\cos(\omega_a t) + \Omega_b\cos(\omega_b t)) 
\end{align}$$

with each multiplied by a Gaussian envelope:

$$\begin{align}
g(t) = e^{-\frac{(t-t_g/2)^2}{2\sigma^2}} - e^{-\frac{(t_g/2)^2}{2\sigma^2}}
\end{align}$$

with $\sigma = t_g/4$ (i.e. $1/4$ of gate time). This gives us a final drive of:

$$\begin{align}
f(t) = (e^{-\frac{(t-t_g/2)^2}{2(t_g/4)^2}} - e^{-\frac{(t_g/2)^2}{2(t_g/4)^2}})(\Omega_a\cos(\omega_a t) + \Omega_b\cos(\omega_b t)) 
\end{align}$$


"""

# ╔═╡ fb425471-dda6-4868-9992-a1eabb7e198b
begin
	envelope(t, tgate, sigma, drive_amp) = drive_amp*(exp(-(t-tgate/2)^2/(2*sigma^2))-exp(-(tgate/2)^2/(2*sigma^2)))
	drive(t, t_gate, ωd1, ωd2, amp1, amp2) = envelope(t,t_gate, t_gate/4, amp1)*cos(ωd1*t) + envelope(t, t_gate, t_gate/4, amp2)*cos(ωd2*t)
end

# ╔═╡ f11226b4-eb16-4fde-bcb4-258ff689be25
H_full.operators[2].data[3,1], H_full.operators[2].data[3,2] 

# ╔═╡ 33d5fca8-6cb8-42a1-bf48-eb53be814eb2
begin
	
	delta = 0.003*c
	w09 = ev[10]-ev[1] - delta
	w29 = ev[10]-ev[3] - delta
	scale = 2
	amp1 = abs(0.005*c/H_full.operators[2].data[3,1])/scale
	amp2 = abs(0.005*c/H_full.operators[2].data[3,2])/scale
	t_gate = 2000
	function Ht(t, psi)
		H_full.factors[2] = drive(t, t_gate, w09, w29, amp1, amp2)
		return H_full
	end
	states = [Ket(trunc_basis, one_vec(length(to_show), n)) for n in range(1,length(to_show))]
	w09, w29, amp1, amp2
end

# ╔═╡ dd309198-db95-4565-9cdf-1a2bf73efbd7
begin
	times = range(0, t_gate, 1000)
	tout, psit = timeevolution.schroedinger_dynamic(times, states[1], Ht, maxiters=10^8)
	psit = [normalize(x) for x in psit]
	proj1 = tensor(states[1], dagger(states[1]))
	plt.close("all")
	for i in range(1,length(to_show))
		plt.plot(tout, real(expect(psi_to_proj(states[i]), psit)), label=L"|"*string(to_show[i]-1)*L"$\rangle$")
	end
	plt.xlabel("time (ns)")
	plt.ylabel("population")
	plt.legend()
	plt.gcf()
end

# ╔═╡ 5f68d6cc-19ac-4f7e-828a-ccf981858bae
begin
	plt.close("all")
	plt.plot(tout, drive.(tout, t_gate, w09, w29, amp1, amp2))
	plt.xlabel("time (ns)")
	plt.ylabel("drive signal")
	plt.gcf()
end

# ╔═╡ 8bc038e1-4761-430d-a7b9-30c3eac29246
md"""
#### Parameters to produce above time evolution:
Given:

$$$H = H_0 + f(t)H_{drive}$$$

the parameters are:
"""

# ╔═╡ a979c60b-de54-427b-9c5f-21b0d3498227
md"""
##### $H_0$:
"""

# ╔═╡ 9f677561-cec4-4818-b156-ac25a9d63ba9
round.(H_full.operators[1].data, digits=3)

# ╔═╡ 5619c30f-e939-4e33-9358-ea657315d55b
md"""
##### $$H_{drive}$$:
"""

# ╔═╡ 15994e01-aeb9-4b35-b350-974a65ffd15f
round.(H_full.operators[2].data, digits=3)

# ╔═╡ d6e52a98-0386-4b29-b774-261a0da2746a
md"""
##### $f(t)$:

$$$f(t) = (e^{-\frac{(t-t_g/2)^2}{2(t_g/4)^2}} - e^{-\frac{(t_g/2)^2}{2(t_g/4)^2}})(\Omega_a\cos(\omega_a t) + \Omega_b\cos(\omega_b t))$$$

$$$\begin{align} 
t_g &= 2000 \, ns \\
\Omega_a/2\pi &= 0.268 \, GHz \\
\omega_a/2\pi &= 50.880 \, GHz \\
\Omega_b/2\pi &= 0.457 \, GHz \\
\omega_b/2\pi &= 29.501 \, GHz

\end{align}$$$

"""

# ╔═╡ c8dee4ed-af2e-43eb-b021-d266b1a7c655
t_gate, amp1, amp2, w09, w29, 11

# ╔═╡ Cell order:
# ╠═e195097a-f66f-11ed-3515-5bdb3424945b
# ╠═d1559695-caab-4f27-a9d6-f0fb8eda7adf
# ╟─a493192e-9026-4cc5-a270-9a7c092ef1de
# ╠═6ab4542b-da76-4ac9-8408-a319f4aebc7d
# ╠═e219c5b2-7c19-41e7-96bf-f0099c5bbfb5
# ╟─045b30e1-e9ef-4df7-8d8a-20c979fecb1a
# ╠═a488bc91-d674-44e5-a544-7a747ca1fc01
# ╟─c6e96ae2-df23-4a58-9c93-46e5720d8c3a
# ╟─5f7f7b2f-cf2a-489a-bc2b-1561b6d0e0e6
# ╠═48f125e4-7c86-49bd-9fc2-e1262b86abe8
# ╟─7d8c18e2-7028-4de9-a0e6-e98715f3b78e
# ╠═5bc6e7b8-0fd8-40f2-b7d4-0603edcc3c62
# ╠═e094238f-aa9c-48e7-9235-d9dbd5e3b8f6
# ╟─18c0cf62-fd9e-46f8-9fc9-a3aedde69f0c
# ╠═4a3d3de2-64d5-4f91-b8f5-29b47e5527fa
# ╠═d521009d-53bf-4430-9ab2-1f45e6c2eae8
# ╟─b645f16b-5588-43c6-a7ed-aa92cf370f00
# ╠═877fb0cf-e3aa-4c89-94f8-d664a6d3a57c
# ╟─2e94f431-5451-488d-ab8e-71a2f3e45b45
# ╠═fb425471-dda6-4868-9992-a1eabb7e198b
# ╟─f11226b4-eb16-4fde-bcb4-258ff689be25
# ╠═33d5fca8-6cb8-42a1-bf48-eb53be814eb2
# ╠═dd309198-db95-4565-9cdf-1a2bf73efbd7
# ╠═5f68d6cc-19ac-4f7e-828a-ccf981858bae
# ╟─8bc038e1-4761-430d-a7b9-30c3eac29246
# ╟─a979c60b-de54-427b-9c5f-21b0d3498227
# ╠═9f677561-cec4-4818-b156-ac25a9d63ba9
# ╟─5619c30f-e939-4e33-9358-ea657315d55b
# ╠═15994e01-aeb9-4b35-b350-974a65ffd15f
# ╟─d6e52a98-0386-4b29-b774-261a0da2746a
# ╠═c8dee4ed-af2e-43eb-b021-d266b1a7c655
