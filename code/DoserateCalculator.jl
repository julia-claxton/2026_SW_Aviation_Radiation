using Statistics, LinearAlgebra          # Core math
using BenchmarkTools, Profile, TickTock  # Debugging
using NPZ, DelimitedFiles                # File interactions
using BasicInterpolators
using Plots, Plots.PlotMeasures
using Glob
include("./General_Functions.jl")
include("./SpectrumParser.jl")
include("./AMS-02Reader.jl")
include("./ARMASReader.jl")

const TOP_LEVEL = dirname(@__DIR__)

function get_doserate_from_ams02(date; results_dir = "$(TOP_LEVEL)/data/GLYPHS", extrapolate_μ_en = true)
    proton_dose = ams02_doserate_species_specific("proton", date, results_dir = results_dir, extrapolate_μ_en = extrapolate_μ_en)
    alpha_dose = ams02_doserate_species_specific("alpha", date, results_dir = results_dir, extrapolate_μ_en = extrapolate_μ_en)

    return Dict(
        "altitude_km" => proton_dose["altitude_km"],
        "doserate_uGy/hr" => proton_dose["doserate_uGy/hr"] + alpha_dose["doserate_uGy/hr"]
    )
end

function ams02_doserate_species_specific(species, date; results_dir = "$(TOP_LEVEL)/data/GLYPHS", extrapolate_μ_en = true)
    # Get beam data
    beam_particles, beam_energies_keV = get_beams(results_dir, input_particle = species)

    # Calculate doserate
    spectrum_energy_min_keV, spectrum_energy_max_keV, spectrum = gcr_spectrum(species, date = date, factor = 1)
        # Input units: keV
        # Output units: # / (m2 s keV)

    beam_weights = energy_spectrum_to_beam_weights(spectrum, spectrum_energy_min_keV, spectrum_energy_max_keV, beam_energies_keV)
        # Units: # / (m2 s)

    flux = beam_weights_to_flux(beam_particles, beam_energies_keV, beam_weights; results_dir = results_dir)
        # Units: # / (m2 s)    

    dose = flux_to_doserate(flux, results_dir = results_dir, extrapolate_μ_en = extrapolate_μ_en)
        # Units: μGy/hr
    return dose
end

function flux_to_doserate(flux_data; results_dir = "$(TOP_LEVEL)/data/GLYPHS", extrapolate_μ_en = true)
    # Get dummy data for result allocation
    beam_particles, beam_energies_keV = get_beams(results_dir)
    dummy_data = get_spectra(results_dir, beam_particles[begin], beam_energies_keV[begin])

    # Loop over species and sum up doserate
    doserate = zeros(length(dummy_data["altitude_km"]))
    for species in ["proton", "electron", "alpha", "gamma"] 
        # Get stopping power
        energy_min_keV, energy_max_keV, stopping_power = get_stopping_power(species, extrapolate_μ_en = extrapolate_μ_en)
            # Input units: keV
            # Output units: keV m2 / kg
        valid_energy_mask_keV = energy_min_keV .≤ dummy_data["energy_bins_mean_keV"] .≤ energy_max_keV
        stopping_power = stopping_power.(dummy_data["energy_bins_mean_keV"][valid_energy_mask_keV])
            # Units: keV m2 / kg

        # Add doserate contribution from this species
        doserate .+= flux_data["$(species)_flux"][:, valid_energy_mask_keV] * stopping_power
            # Units: [# / (m2 s)] • [keV m2 / (# kg_silicon)] = keV / (s kg_silicon)
    end

    # Convert doserate to Gy/hr. 1 Gy = 1 J/kg
    doserate .*= (1e3) .* (1.6e-19) .* 3600 # In order, 1000 ev/keV, 1.6e-19 J/eV, 3600 s/hr
        # Units: J / (kg_silicon hr) = Gy/hr

    # Convert doserate to μGy/hr
    doserate .*= 1e6

    return Dict(
        "altitude_km" => dummy_data["altitude_km"],
        "doserate_uGy/hr" => doserate
    )
end

function beam_weights_to_flux(beam_particles, beam_energies_keV, beam_fluxes; results_dir = "$(TOP_LEVEL)/data/GLYPHS")
    # Get dummy data for result allocation
    dummy_particles, dummy_energies = get_beams(results_dir)
    dummy_data = get_spectra(results_dir, dummy_particles[begin], dummy_energies[begin])

    # Allocate results
    species_to_tally = ["proton", "electron", "gamma", "alpha"]
    result = Dict{String, Any}()
    result["altitude_km"] = dummy_data["altitude_km"]
    result["energy_bins_mean_keV"] = dummy_data["energy_bins_mean_keV"]
    [result["$(species)_flux"] = zeros(size(dummy_data["proton_counts"])) for species in species_to_tally]
        
    # Loop over all input energies we are given and sum up flux at each altitude and energy bin for that beam
    for beam_idx in eachindex(beam_energies_keV)
        data = get_spectra(results_dir, beam_particles[beam_idx], beam_energies_keV[beam_idx])
        for species in species_to_tally
            result["$(species)_flux"] .+= beam_fluxes[beam_idx] .* data["$(species)_counts"] ./ data["n_particles"]
        end
    end
    return result
end

function energy_spectrum_to_beam_weights(input_spectrum, lower_bound_keV, upper_bound_keV, beam_energies_keV)
    # Guard
    if issorted(beam_energies_keV) == false; @warn "This probably won't work with unsorted beam energies"; end

    # Get energy span each beam is responsible for covering in logspace
    energy_midpoints = 10.0 .^ [mean([log10(beam_energies_keV[i]), log10(beam_energies_keV[i+1])]) for i in 1:length(beam_energies_keV)-1]

    # Get half-bin size of start/stop beams so we can append that to the start/stop of the edges
    edges_start = 10.0 ^ ( log10(beam_energies_keV[begin]) - (log10(energy_midpoints[begin]) - log10(beam_energies_keV[begin])) )
    edges_stop  = 10.0 ^ ( log10(beam_energies_keV[end])   + (log10(beam_energies_keV[end])  - log10(energy_midpoints[end]))    )

    # Construct energy edges
    energy_bin_edges = [edges_start, energy_midpoints..., edges_stop]

    # Find the input beams that lie within valid range for the spectrum
    valid_beams = lower_bound_keV .≤  beam_energies_keV .≤ upper_bound_keV

    # Integrate the input spectrum over each beam's span to get beam flux with energy unit integrated out
    energy_bin_starts = clamp.(
        energy_bin_edges[begin:end-1][valid_beams],
        lower_bound_keV,
        upper_bound_keV
    )
    energy_bin_stops = clamp.(
        energy_bin_edges[begin+1:end][valid_beams],
        lower_bound_keV,
        upper_bound_keV
    )
    
    # Trapezoidal integration
    ΔE = energy_bin_stops .- energy_bin_starts # Units: keV
    beam_fluxes = [(1/2) * ΔE[i] * (input_spectrum(energy_bin_stops[i]) + input_spectrum(energy_bin_starts[i])) for i in eachindex(ΔE)]
    
    # Construct beam weight vector
    beam_weights = zeros(length(beam_energies_keV))
    beam_weights[valid_beams] = beam_fluxes

    return beam_weights
end

function gcr_spectrum(species; date = nothing, factor = 1)
    # Get AMS-02 data
    data = get_ams02_spectrum(date = date)

    # Do interpolation in logspace because the interpolator gets a little weird when done in linspace
    gcr_logspace_interpolator = LinearInterpolator(data["$(species)_energy_bins_mean_GeV"] .* 1e6, log10.(data["$(species)_flux"] ./ 1e6)) # Multiply/divide by 1e6 to convert all GeV units to keV
    
    # Get final quantities and return
    spectrum_energy_min_keV = gcr_logspace_interpolator.r.xa
    spectrum_energy_max_keV = gcr_logspace_interpolator.r.xb
    spectrum(E) = factor * 2π * (10.0 ^ gcr_logspace_interpolator(E)) # Multiply by 2π = assuming uniform incidence over 2π steradian hemisphere
        # Input units: keV
        # Output units: # / (m2 s keV)

    return spectrum_energy_min_keV, spectrum_energy_max_keV, spectrum
end

function get_stopping_power(incident_particle; extrapolate_μ_en = true)
    # Special behavior for photons since energy deposition is reported differently for gammas
    if incident_particle == "gamma"; return photon_energy_deposition(extrapolate_μ_en = extrapolate_μ_en); end
    
    if incident_particle ∉ ["proton", "electron", "alpha"]
        error("Argument \"$(incident_particle)\" not recognized.")
    end

    # Get stopping power data in silicon
    data = readdlm("$(TOP_LEVEL)/data/Silicon_$(incident_particle)_Stopping_Power.txt", ' ', skipstart = 8)
    input_energy_MeV = data[:,1]   # Units: MeV
    stopping_power_raw = data[:,2] # Units: MeV cm2 / g

    # Change the units to be compatible with the rest of our calculations
    input_energy_keV = input_energy_MeV .* 1e3 # Units: keV
    stopping_power_data = stopping_power_raw .* (1e3) * (1e3) * (1/100)^2 # Units: keV m2 / kg
        # In order: 1000 keV/MeV, 1000 g/kg, 100 cm/m

    logspace_interpolator = LinearInterpolator(input_energy_keV, log10.(stopping_power_data))
    stopping_power(E) = 10.0 ^ logspace_interpolator(E) # Units: keV m2 / kg

    return input_energy_keV[begin], input_energy_keV[end], stopping_power # Units: keV m2 / kg
end

function photon_energy_deposition(; extrapolate_μ_en = true)
    data = readdlm("$(TOP_LEVEL)/data/Silicon_xray_mass_attenuation.txt", ' ', skipstart = 11)
    input_energy_MeV = data[:,1]   # Units: MeV
    μ_en = data[:, 3] # Mass Energy-Absorption Coefficient. Units cm2 / g

    # There are two data points at the same energy to show K shell ionization discontinuity. We need to nudge one
    # of them for our interpolator to play nice with it.
    @assert input_energy_MeV[3] == input_energy_MeV[4]
    input_energy_MeV[3] -= 1e-8
    
    # Change the units to be compatible with the rest of our calculations
    input_energy_keV = input_energy_MeV .* 1e3 # Units: keV

    # Extrapolate above maximum provided energy
    if extrapolate_μ_en == true
        extrapolation_energies = logrange(1.1 * input_energy_keV[end], 1e6*input_energy_keV[end], 100)
        input_energy_keV = [input_energy_keV..., extrapolation_energies...]
        μ_en = [μ_en..., repeat([μ_en[end]], length(extrapolation_energies))...]
    end

    # Get energy deposition
    energy_deposition = μ_en .* input_energy_keV .* (1e3) * (1/100)^2 # Units: keV m2 / kg
        # [cm2 / g] * [keV] * [1000 g/kg] * [1/100 m/cm]^2

    logspace_interpolator = LinearInterpolator(input_energy_keV, log10.(energy_deposition))
    stopping_power(E) = 10.0 ^ logspace_interpolator(E) # Units: keV m2 / kg

    return input_energy_keV[begin], input_energy_keV[end], stopping_power # Units: keV m2 / kg
end

function bethe_formula_silicon(name, _KE)
    # Convert input energy from keV to J
    KE = _KE * 1000   # Units: eV
    KE = KE * 1.6e-19 # Units: J

    # Constants
    mₑ = 9.11e-31  # Electron mass, kg
    qₑ = 1.602e-19 # Elementary charge, C
    mₚ = 1.67e-27  # Proton mass, kg
    c = 299792458  # Speed of light, m/s
    Nₐ = 6.022e23  # Avagadro's number, #/mol
    Mᵤ = 1e-3      # Molar mass constant, kg/mol
    ϵ₀ = 8.854e-12 # Permittivity of free space, F/m

    # Incident particle properties
        # m = Particle mass, kg
        # z = Particle charge number, unitless
    if name == "proton";   m = mₚ; z = 1; end
    if name == "electron"; m = mₑ; z = 1; end
    v = c * (1 - (((KE/(m*c^2)) + 1)^(-2)))^(1/2) # Incident particle velocity, m/s 
    
    # Target properties: Silicon (Si, 14)
    Z = 14                        # Target atomic number, unitless
    A = 28.085                    # Relative atomic mass, unitless
    ρ = 2.329085 * (100^3) / 1000 # Target mass density, kg/m³
    I = 173.000000 * 1.6e-19      # Mean excitation energy, J. Source: https://physics.nist.gov/cgi-bin/Star/compos.pl?matno=014

    # Intermediate quantities
    β = v/c # Unitless
    n = Nₐ * Z * ρ / (A * Mᵤ) # Target electron density, #/m³

    # Terms
    term1 = 4π / (mₑ * c^2)                                       # Units: J⁻¹
    term2 = n * (z^2) / (β^2)                                     # Units: m⁻³
    term3 = ( (qₑ^2) / (4 * π * ϵ₀) )^2                           # Units: J² m²
    term4 = log( (2*mₑ*(c^2)*(β^2)) / (I * (1 - (β^2))) ) - (β^2) # Unitless

    # Final result
    dEdx = term1 * term2 * term3 * term4
        # Units: J⁻¹ m⁻³ J² m² = J/m

    # Unit conversion to keV cm² g⁻¹
    dEdx = dEdx / ρ         # Units: J m² kg⁻¹
    dEdx = dEdx * (100^2)   # Units: J cm² kg⁻¹
    dEdx = dEdx / (1000)    # Units: J cm² g⁻¹
    dEdx = dEdx / (1.6e-19) # Units: eV cm² g⁻¹
    dEdx = dEdx / (1e3)     # Units: keV cm² g⁻¹

    # we are 10x too large precisely. where???
    @warn "electrons are crabbed and protons are good but 10x too big"

    return dEdx # Units: keV cm² g⁻¹
end

function plot_stopping_power(incident_particle)
    energy_min_keV, energy_max_keV, stopping_power = get_stopping_power(incident_particle)
    energy = logrange(energy_min_keV, energy_max_keV, 1000)

    to_plot = stopping_power.(energy)
    ymin_logspace = floor(log10(min(to_plot...)))
    ymax_logspace = ceil(log10(max(to_plot...)))
    plot(energy, to_plot,
        title = "$(uppercase(incident_particle[1]))$(incident_particle[2:end]) Energy Deposition in Si",
    
        label = false,
        linecolor = :black,
        linewidth = 2,

        xlabel = "Energy, keV",
        xlims = (energy_min_keV, energy_max_keV),
        xscale = :log10,
        xminorgrid = true,

        ylabel = "Energy Deposition, keV m2 / kg_Si",
        ylims = 10.0 .^ (ymin_logspace, ymax_logspace),
        yticks = 10.0 .^ (ymin_logspace:ymax_logspace),
        yscale = :log10,
        yminorgrid = true,

        framestyle = :box,
        tickdirection = :out,
        size = (1,1) .* 450
    )
    box_aspect!(1)
    display(plot!())
end

function plot_gcr_coverage(; results_dir = "$(TOP_LEVEL)/data/GLYPHS")
    beam_particles, beam_energies = get_beams(results_dir)
    mask = beam_particles .== "proton"

    energy_min_keV, energy_max_keV, spectrum = gcr_spectrum()
    energy = logrange(energy_min_keV, energy_max_keV, 1000)

    xmin_log = floor(log10(min(beam_energies..., energy_min_keV)))
    xmax_log = ceil(log10(max(beam_energies..., energy_max_keV)))
    plot(
        xlabel = "Energy, keV",
        xscale = :log10,
        xticks = 10.0 .^ (xmin_log:xmax_log),

        ylabel = "GCR Flux, # / (m2 s GeV)",
        yscale = :log10,
        ylims = 10.0 .^ (-7, -1),
        dpi = 300
    )
    vline!(beam_energies[mask],
        label = false,
        linecolor = :grey,
    )
    plot!(energy, spectrum.(energy),
        label = false,
        linewidth = 2
    )
    display("image/png", plot!())
end

function smooth_gamma_flux!(flux_data; tolerance = .1)
    boundary_idxs = findall( (flux_data["altitude_km"] .% 1) .≤ tolerance )


    heatmap(log10.(flux_data["gamma_flux"]), bg = :black)
    display(plot!())


    flux_data["gamma_flux"][boundary_idxs,:] .= 0

    for altitude_idx in boundary_idxs
        idxs_to_average = [altitude_idx - 1, altitude_idx + 1]

        deleteat!(idxs_to_average, idxs_to_average .≤ 0)
        deleteat!(idxs_to_average, idxs_to_average .> length(flux_data["altitude_km"]))

        flux_data["gamma_flux"][altitude_idx, :] .= [mean(flux_data["gamma_flux"][idxs_to_average, E]) for E in eachindex(flux_data["energy_bins_mean_keV"])]
    end

    heatmap(log10.(flux_data["gamma_flux"]), bg = :black)
    display(plot!())
    error("This sucks. Fix it.")
end

#bethe_formula_silicon("electron", 1000)
#_, _, sp = get_stopping_power("electron")