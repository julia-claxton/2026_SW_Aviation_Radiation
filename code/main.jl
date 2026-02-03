using Statistics, LinearAlgebra          # Core math
using BenchmarkTools, Profile, TickTock  # Debugging
using NPZ, DelimitedFiles                # File interactions
using BasicInterpolators
using Plots, Plots.PlotMeasures
    default(
        dpi = 300,
        framestyle = :box,
        tickdirection = :out,
        label = false
    )
using Glob
using Base.Threads
include("./General_Functions.jl")
include("./DoserateCalculator.jl")
include("./Julia_ELFIN_Tools/Events.jl")
include("./Julia_ELFIN_Tools/Visualization.jl")

const TOP_LEVEL = dirname(@__DIR__)

function save_example_energy_histograms()
    E = 1_114_242.9 # Beam energy, keV
    data = get_spectra(results_dir, "proton", E)
    delete!(data, "input_particle")
    npzwrite("$(TOP_LEVEL)/data/figure_data/example_histograms.npz", data)
end

function save_elfin_example_event()
    # Use 99.9th percentile event
    event = create_event(
        DateTime("2022-02-04T15:17:32.015"), DateTime("2022-02-04T15:17:33"), "B",
        data_type = "hs",
        maximum_relative_error = 0.5
    )
    npzwrite("$(TOP_LEVEL)/data/figure_data/elfin_event.npz",
        n_flux = event.n_flux[1,:,:],
        energy_bins_mean_kev = event.energy_bins_mean,
        pitch_angles = event.pitch_angles[1,:],
        loss_cone_angle = event.loss_cone_angles[1],
        anti_loss_cone_angle = event.anti_loss_cone_angles[1]
    )
end

function save_gcr_spectrum()
    ams02_coverage = get_ams02_coverage()
    spectrum = get_ams02_spectrum(date = ams02_coverage[1000])
    dose = get_doserate_from_ams02(ams02_coverage[1000])

    #@show ams02_coverage[1000]

    npzwrite("$(TOP_LEVEL)/data/figure_data/gcr_spectrum.npz", merge(spectrum, dose))
end

function save_stopping_power()
    proton_min_kev, proton_max_kev, proton_stopping_power = get_stopping_power("proton")
    electron_min_kev, electron_max_kev, electron_stopping_power = get_stopping_power("electron")
    alpha_min_kev, alpha_max_kev, alpha_stopping_power = get_stopping_power("alpha")
    photon_min_kev, photon_max_kev, photon_stopping_power = get_stopping_power("gamma")

    proton_energy_kev = collect(logrange(proton_min_kev, proton_max_kev, 1000))
    electron_energy_kev = collect(logrange(electron_min_kev, electron_max_kev, 1000))
    alpha_energy_kev = collect(logrange(alpha_min_kev, alpha_max_kev, 1000))
    photon_energy_kev = collect(logrange(photon_min_kev, 20e3, 1000)) # 20e3 = end of tables, afterwards = extrapolation
    photon_energy_extrapolate_kev = collect(logrange(20e3, photon_max_kev, 10))

    npzwrite("$(TOP_LEVEL)/data/figure_data/stopping_power.npz",
        Dict{String, Vector{Float64}}(
            "proton_energy_kev" => proton_energy_kev,
            "proton_stopping_power" => proton_stopping_power.(proton_energy_kev),

            "electron_energy_kev" => electron_energy_kev,
            "electron_stopping_power" => electron_stopping_power.(electron_energy_kev),

            "alpha_energy_kev" => alpha_energy_kev,
            "alpha_stopping_power" => alpha_stopping_power.(alpha_energy_kev),

            "photon_energy_kev" => photon_energy_kev,
            "photon_mu_en" => photon_stopping_power.(photon_energy_kev) ./ photon_energy_kev,

            "photon_energy_extrapolated_kev" => photon_energy_extrapolate_kev,
            "photon_mu_en_extrapolated" => photon_stopping_power.(photon_energy_extrapolate_kev) ./ photon_energy_extrapolate_kev
        )
    )
end

function save_example_armas_data()
    # Save ARMAS datasets we want to plot in the paper
    data_names = ["agree", "nairas_under_1", "nairas_under_2", "excess", "presentation"]
    start_times_to_save = [DateTime("2015-12-19T00:59:40"), DateTime("2019-09-14T10:40:20"), DateTime("2019-09-05T17:15:20"), DateTime("2017-10-14T21:08:10"), DateTime("2015-10-03T11:31:10")]
    vehicles_to_save = ["AFRC", "PN-2", "DC87", "DC-8", "G-V"]
    assets_to_save = ["FM1001", "FM6002", "FM5003", "FM1001", "FM2002"]

    to_save = Dict{String, Any}()
    for i in eachindex(vehicles_to_save)
        data = get_armas_data(start_times_to_save[i], vehicles_to_save[i], assets_to_save[i])
        altitude, nairas, armas, excess = get_armas_doserate_data(data) 
        glyphs_prediction = get_doserate_from_ams02(Date(start_times_to_save[i]))
        glyphs_prediction_no_extrap = get_doserate_from_ams02(Date(start_times_to_save[i]), extrapolate_μ_en = false)

        to_save["$(data_names[i])_altitude"] = altitude
        to_save["$(data_names[i])_nairas"] = nairas
        to_save["$(data_names[i])_armas"] = armas
        to_save["$(data_names[i])_excess"] = excess
        to_save["$(data_names[i])_glyphs_altitude"] = glyphs_prediction["altitude_km"]
        to_save["$(data_names[i])_glyphs_doserate"] = glyphs_prediction["doserate_uGy/hr"]
        to_save["$(data_names[i])_glyphs_doserate_no_extrap"] = glyphs_prediction_no_extrap["doserate_uGy/hr"]
    end
    npzwrite("$(TOP_LEVEL)/data/figure_data/example_armas_data.npz", to_save)
end

function get_armas_doserate_data(armas_data)
    armas_altitude  = armas_data["alt(GPS)"] ./ 1e3 # Units: km
    nairas_doserate = armas_data["NL2DrSi"]         # Units: μGy hr⁻¹
    armas_doserate  = armas_data["AL2DrSi"]         # Units: μGy hr⁻¹

    # Get doserate difference between ARMAS and NAIRAS
    excess_doserate = armas_doserate .- nairas_doserate # Units: μGy hr⁻¹

    # Trim down to conditions we want to look at
    mask = armas_data["data_valid"] .== true # Valid data
    mask = mask .&& (0 .≤ armas_altitude .≤ 30) # Aviation altitude range
    #mask = mask .&& (armas_data["Lshell"] .≤ 3) # Consistency

    return armas_altitude[mask], nairas_doserate[mask], armas_doserate[mask], excess_doserate[mask]
end

function save_elfin_derived_doserates()
    labels = ["top", "percentile_98", "percentile_90"]
    start_times = [DateTime("2022-02-04T15:17:32.015"), DateTime("2021-11-07T02:12:46.867"), DateTime("2020-10-22T23:04:22.315")]
    satellites = ["B", "B", "A"]

    results = Dict{String, Any}()
    for i in eachindex(labels)
        dose_data = elfin_doserate(start_times[i], satellites[i])
        results["$(labels[i])_doserate_uGy/hr"] = dose_data["doserate_uGy/hr"]
        results["$(labels[i])_energy_spectrum"] = dose_data["energy_spectrum"]
        
        results["altitude_km"] = dose_data["altitude_km"]
        results["energy_labels"] = dose_data["energy_labels"]
    end

    npzwrite("$(TOP_LEVEL)/data/figure_data/elfin_electron_doserate.npz", results)
    return results
end

function elfin_doserate(start_time, satellite)
    event = create_event(start_time, start_time + Second(2), satellite, maximum_relative_error = 0.5, data_type = "hs")

    α = event.pitch_angles[1, :]
    α_lc = event.loss_cone_angles[1]
    n_flux = event.n_flux[1, :, :]
        # Units # cm⁻² s⁻¹ MeV⁻¹ str⁻¹

    # Trim to only precipitating flux
    if α_lc < 90 # Northern hemisphere
        precipitating_n_flux = n_flux[:, α .≤ α_lc]
    else # Southern hemisphere
        precipitating_n_flux = n_flux[:, α .≥ α_lc]
    end
        # Units # cm⁻² s⁻¹ MeV⁻¹ str⁻¹

    # Integrate out solid angle
    precipitating_n_flux .*= ELFIN_EPD_SOLID_ANGLE
        # Units # cm⁻² s⁻¹ MeV⁻¹

    # Integrate out energy
    ΔE = (event.energy_bins_max .- event.energy_bins_min) ./ 1000 # Units: MeV
    precipitating_n_flux = [precipitating_n_flux[E, α] * ΔE[E] for E in 1:16, α in 1:size(precipitating_n_flux)[2]]
        # Units # cm⁻² s⁻¹

    # Rebin
    energy_spectrum = dropdims(sum(precipitating_n_flux, dims = 2), dims = 2)
        # Units # cm⁻² s⁻¹

    # Convert units
    energy_spectrum .*= 100^2
        # Units # m⁻² s⁻¹

    # Calculate doserate
    results_dir = "$(TOP_LEVEL)/data/GLYPHS"
    beam_particles, beam_energies_keV = get_beams(results_dir, input_particle = "electron")

    elfin_energies = round.(event.energy_bins_mean)
    beam_weights = zeros(length(beam_energies_keV))
        # Units: # m⁻² s⁻¹
    for i in eachindex(beam_weights)
        idx = findfirst(beam_energies_keV[i] .== elfin_energies)
        if idx == nothing; continue; end

        beam_weights[i] = energy_spectrum[idx]
    end

    flux = beam_weights_to_flux(beam_particles, beam_energies_keV, beam_weights; results_dir = results_dir)
        # Units: # m⁻² s⁻¹

    dose = flux_to_doserate(flux, results_dir = results_dir)
    # Units: μGy/hr

    # Add in intermediate info for plotting
    dose["energy_spectrum"] = energy_spectrum
    dose["energy_labels"] = event.energy_bins_mean

    #@show event.time_datetime[start_idx]
    #@show event.satellite

    return dose
end

function find_elfin_spectra()
    keep_top = 500 # Keep the top N hardest spectra

    elfin_dates, elfin_sats = all_elfin_science_dates_and_satellite_ids()

    # Results arrays
    labels = ["top", "percentile_98", "percentile_90"]

    start_times = Dict()
    sats = Dict()
    supra_MeV_flux = Dict()

    for label in labels
        start_times[label] = fill(DateTime(0), keep_top)
        sats[label] = fill("", keep_top)
        supra_MeV_flux[label] = fill(0.0, keep_top)
    end

    hardness_bin_edges = 10.0 .^ (0:.01:9)
    hardness_distribution = zeros(length(hardness_bin_edges)-1) # For finding approximate percentile of some hardness value

    # Locks
    writelock = ReentrantLock()
    
    # Loop
    n_complete = 0
    threshold = 0.0
    Threads.@threads for i in eachindex(elfin_dates)
        event = create_event(elfin_dates[i], elfin_sats[i], data_type = "hs", maximum_relative_error = 0.5);
        if event == nothing; n_complete += 1; continue; end
        if event.data_reliable == false; n_complete += 1; continue; end

        # Calculate hardness of each half spin
        hardness = [get_supra_MeV_flux(event, i) for i in 1:event.n_datapoints]
        
        lock(writelock) do
            n_complete += 1

            # Grab hardest spectra
            add_to_leaderboard!(start_times["top"], sats["top"], supra_MeV_flux["top"], hardness, event.satellite, event.time_datetime,
                hardness .> threshold
            )
            threshold = supra_MeV_flux["top"][end]

            # Grab 98th percentile (which is around 3.5e3)
            add_to_leaderboard!(start_times["percentile_98"], sats["percentile_98"], supra_MeV_flux["percentile_98"], hardness, event.satellite, event.time_datetime,
                3.5e3 .≤ hardness .≤ 3.6e3
            )
            
            # Grab 90th percentile (which is around 8.5e2)
            add_to_leaderboard!(start_times["percentile_90"], sats["percentile_90"], supra_MeV_flux["percentile_90"], hardness, event.satellite, event.time_datetime,
                8.4e2 .≤ hardness .≤ 8.5e2
            )

            # Add to pdf
            hardness_distribution .+= exact_1dhistogram(hardness, hardness_bin_edges)

            # Progress update
            print_progress_bar(n_complete/length(elfin_dates))
        end
    end

    # Write spectrum data
    for label in labels
        header = ["start_time" "sat" "supra_MeV_flux"]
        data = cat(start_times[label], sats[label], supra_MeV_flux[label], dims = 2)
        writedlm("$(TOP_LEVEL)/results/$(label)_elfin_spectra.csv", cat(header, data, dims = 1), ',')
    end

    # Write hardness pdf & cdf
    npzwrite("$(TOP_LEVEL)/results/hardness_distribution.npz",
        bin_edges = hardness_bin_edges,
        pdf = hardness_distribution,
        cdf = cumsum(hardness_distribution)
    )
end

function add_to_leaderboard!(result_start_times, result_sats, result_flux, event_hardness, event_sat, event_time, condition)
    for i in findall(condition)
        place = findfirst(event_hardness[i] .> result_flux)
        if place == nothing; continue; end

        result_flux[:] = [result_flux[1:place-1]..., event_hardness[i], result_flux[place:end-1]...]
        result_sats[:] = [result_sats[1:place-1]..., event_sat, result_sats[place:end-1]...]
        result_start_times[:] = [result_start_times[1:place-1]..., event_time[i], result_start_times[place:end-1]...]
    end
end

function get_supra_MeV_flux(event::Event, idx)
    # Integrate out solid angle
    n_flux = event.n_flux[idx, :, :] .* ELFIN_EPD_SOLID_ANGLE
        # Units: # MeV⁻¹ cm⁻² s⁻¹

    # Move to northern hemisphere for consistency
    α = copy(event.pitch_angles[idx, :])
    α_lc = event.loss_cone_angles[idx]
    if α_lc > 90
        n_flux = reverse(n_flux, dims = 2)
        α_lc = 180 - α_lc 
        α = reverse(180 .- α)
    end

    # Cut out non-precipitating counts
    trimmed_n_flux = copy(n_flux)
    trimmed_n_flux[:, α .> α_lc] .= 0

    # Sum to energy spectrum
    energy_spectrum = dropdims(sum(trimmed_n_flux, dims = 2), dims = 2)
        # Units: # MeV⁻¹ cm⁻² s⁻¹

    # Integrate out energy
    ΔE = (event.energy_bins_max .- event.energy_bins_min) ./ 1000
    energy_spectrum .*= ΔE
        # Units: # cm⁻² s⁻¹

    return sum(energy_spectrum[10:end])  # Units: # cm⁻² s⁻¹
end

function save_conjunction_data(; results_dir = "$(TOP_LEVEL)/data/GLYPHS", show_plot = false)
    # Find AMS-02 and ARMAS conjunctions
    armas_coverage = get_armas_coverage()
    ams02_coverage = get_ams02_coverage()
    conjunction_idxs = findall([Date(el) in ams02_coverage for el in armas_coverage["start_times"]])

    # Result arrays
    residual_bin_edges = -5.25:.5:5.25
    glyphs_residuals = zeros(length(residual_bin_edges)-1)
    nairas_residuals = zeros(length(residual_bin_edges)-1)
    
    altitude_bin_edges = 0:.25:30
    excess_dose_bin_edges = 10.0 .^ (-2:.1:1)
    excess_dose_histogram = zeros(length(altitude_bin_edges)-1, length(excess_dose_bin_edges)-1)

    excess_doserate = Float64[]
    altitude = Float64[]

    # Iterate
    writelock = ReentrantLock()
    n_completed = 0
    print_progress_bar(0)
    Threads.@threads for i in eachindex(conjunction_idxs)
        idx = conjunction_idxs[i]

        # Get data
        date = Date(armas_coverage["start_times"][idx])
        ams_dose_data = get_doserate_from_ams02(date, results_dir = results_dir)
        armas_data = get_armas_data(armas_coverage, idx)
        if armas_data["sep"] ≠ 0; continue; end

        # Get NAIRAS doserate
        valid = armas_data["data_valid"] .== true
        armas_altitude_km = armas_data["alt(GPS)"][valid] ./ 1e3
        nairas_doserate = armas_data["NL2DrSi"][valid]
        armas_doserate = armas_data["AL2DrSi"][valid]
        mask = 0 .≤ armas_altitude_km .≤ 30

        # Get residuals
        gcr_interpolator = LinearInterpolator(ams_dose_data["altitude_km"], ams_dose_data["doserate_uGy/hr"], NoBoundaries())
        gcr_dose = gcr_interpolator.(armas_altitude_km[mask])
        
        glyphs_Δdr = armas_doserate[mask] .- gcr_dose
        nairas_Δdr = armas_doserate[mask] .- nairas_doserate[mask]

        # Get electron fraction of excess dose
        glyphs_gcr = gcr_dose
        nairas = nairas_doserate[mask]
        reference_doserate = bin_doserate_to_armas_resolution(glyphs_gcr)

        Δdr = reference_doserate .- armas_doserate[mask]
        underestimation_altitudes = armas_altitude_km[mask][Δdr .> 0]

        # Write results
        lock(writelock) do
            glyphs_residuals .+= exact_1dhistogram(glyphs_Δdr, residual_bin_edges)
            nairas_residuals .+= exact_1dhistogram(nairas_Δdr, residual_bin_edges)
            
            excess_dose_histogram .+= exact_2dhistogram(underestimation_altitudes, Δdr[Δdr .> 0], altitude_bin_edges, excess_dose_bin_edges)

            append!(altitude, armas_altitude_km[mask][Δdr .> 0])
            append!(excess_doserate, Δdr[Δdr .> 0])

            n_completed += 1
            print_progress_bar(n_completed/length(conjunction_idxs))
        end

        if show_plot == true
            println("$(armas_coverage["start_times"][idx]), $(armas_data["VehID"][1]), $(armas_data["Asset"][1])")
            plot_doserates(ams_dose_data, armas_data)
        end
    end
    npzwrite("$(TOP_LEVEL)/data/figure_data/conjunction_data.npz",
        residual_bin_edges = residual_bin_edges,
        sepiida_residuals = glyphs_residuals,
        nairas_residuals = nairas_residuals,

        altitude_bin_edges = altitude_bin_edges,
        excess_dose_bin_edges = excess_dose_bin_edges,
        excess_dose_histogram = excess_dose_histogram
    )
    npzwrite("$(TOP_LEVEL)/data/figure_data/excess_doserate_datapoints.npz",
        altitude = altitude,
        excess_doserate = excess_doserate
    )
    return
end

function bin_doserate_to_armas_resolution(dose)    
    binned_dose = round_nearest.(dose, 0.12) # uDOS has a resolution of 0.12 μGy
    binned_dose[binned_dose .< 0.84/2] .= 0 # ARMAS data never goes below 0.84 μGy/hr
    binned_dose[0.84/2 .≤ binned_dose .≤ 0.84 ] .= 0.84
    return binned_dose
end

function plot_doserates(ams_dose_data, armas_data)
    valid = armas_data["data_valid"] .== true
    armas_altitude_km = armas_data["alt(GPS)"][valid] ./ 1e3
    nairas_doserate = armas_data["NL2DrSi"][valid]
    armas_doserate = armas_data["AL2DrSi"][valid]

    plot(
        title = Date(armas_data["time"][begin]),

        xlabel = "Doserate, μGy/hr",
        xlims = (0, 5),
        xticks = 0:5,
        xminorticks = 2,
        
        ylabel = "Altitude, km",
        ylims = (0, 15),
        yticks = 0:5:15,
        yminorticks = 5,

        bottommargin = -5mm,
        topmargin = -5mm,

        legendposition = :bottomright,
        minorgrid = true,
        size = (1, 1.5) .* 450
    )
    scatter!(nairas_doserate, armas_altitude_km,
        label = "NAIRAS",
        marker = :blue,
        markeralpha = 1,
        markersize = 4,
        markershape = :x
    )
    scatter!(armas_doserate, armas_altitude_km,
        label = "ARMAS",
        xerr = armas_doserate .* 0.2,
        marker = :black,
        markersize = 3,
    )
    plot!(ams_dose_data["doserate_uGy/hr"], ams_dose_data["altitude_km"],
        label = "AMS-02 + GLYPHS (this work)",
        linecolor = :red,
        linewidth = 2,
    )
    box_aspect!(1.5)
    display(plot!())
    return plot!()
end

function plot_a_gcr_doserate(results_dir)
    ams02_coverage = get_ams02_coverage()
    data = get_doserate_from_ams02(ams02_coverage[1], results_dir = results_dir)

    plot(data["doserate_uGy/hr"], data["altitude_km"],
        linecolor = :black,
        linewidth = 2,

        xlabel = "Doserate, μGy/hr",
        xlims = (0, 4.5),
        xticks = 0:4,
        xminorticks = 2,
        
        ylabel = "Altitude, km",
        ylims = (0, 100),
        yticks = 0:10:100,
        yminorticks = 2,

        minorgrid = true,
        grid = 1,
        size = (1.5,2) .* 300,
    )
    box_aspect!(1.5)
    display("image/png", plot!())
end

function view_elfin_spectra(path)
    data = readdlm(path, ',', skipstart = 1)
    starts = DateTime.(data[:,1])
    sats = data[:,2]
    hardness = data[:,3]

    # Hardest
    if path == "$(TOP_LEVEL)/results/hardest_elfin_spectra.csv"
        @show starts[139]
        @show sats[139]
    end

    # 98th percentile
    if path == "$(TOP_LEVEL)/results/percentile_98_elfin_spectra.csv"
        @show starts[1]
        @show sats[1]
    end

    # 90th percentile
    if path == "$(TOP_LEVEL)/results/percentile_90_elfin_spectra.csv"
        @show starts[1]
        @show sats[1]
    end


    for i in eachindex(starts)
        event = create_event(starts[i], starts[i] + Second(2), sats[i], data_type = "hs", maximum_relative_error = 0.5)
        heatmap(log10.(event.n_flux[1,:,:]), title = i)
        p1 = box_aspect!(1)

        event2 = create_event(starts[i]-Minute(2), starts[i] + Minute(2), sats[i], data_type = "hs", maximum_relative_error = 0.5)
        energy_time_series(event2, show_plot = false, by = "date")
        p2 = plot!(clims = (5,10))

        layout = @layout [a; b{.3h}]
        plot(p1, p2, layout = layout, dpi = 300)
        display(plot!())
    end
end

function fraction_contributed_by_out_of_range_gammas()
    # Get beam data
    beam_particles, beam_energies_keV = get_beams(results_dir, input_particle = "proton")

    # Calculate doserate
    spectrum_energy_min_keV, spectrum_energy_max_keV, spectrum = gcr_spectrum("proton")
        # Input units: keV
        # Output units: # / (m2 s keV)

    beam_weights = energy_spectrum_to_beam_weights(spectrum, spectrum_energy_min_keV, spectrum_energy_max_keV, beam_energies_keV)
        # Units: # / (m2 s)

    proton_induced_flux = beam_weights_to_flux(beam_particles, beam_energies_keV, beam_weights; results_dir = results_dir)
        # Units: # / (m2 s)   
        
        
    # Alpha time
    spectrum_energy_min_keV, spectrum_energy_max_keV, spectrum = gcr_spectrum("alpha")
        # Input units: keV
        # Output units: # / (m2 s keV)

    beam_weights = energy_spectrum_to_beam_weights(spectrum, spectrum_energy_min_keV, spectrum_energy_max_keV, beam_energies_keV)
        # Units: # / (m2 s)

    alpha_induced_flux = beam_weights_to_flux(beam_particles, beam_energies_keV, beam_weights; results_dir = results_dir)
        # Units: # / (m2 s)   

    flux = copy(proton_induced_flux)
    for key in keys(flux)
        if contains(key, "flux") == false; continue; end

        flux[key] = proton_induced_flux[key] + alpha_induced_flux[key]
    end

    # Dose time
    dose = flux_to_doserate(flux, results_dir = results_dir)
        # Units: μGy/hr

    extrap_gamma = copy(dose["doserate_uGy/hr"])


    idxs_to_kill = flux["energy_bins_mean_keV"] .≥ 20e3
    flux["gamma_flux"][:,idxs_to_kill] .= 0

    dose = flux_to_doserate(flux, results_dir = results_dir)
    # Units: μGy/hr

    ignore_gamma = copy(dose["doserate_uGy/hr"])

    plot(
        xlims = (0,4),
        xticks = 0:.2:5,
        ylims = (0, 30)
    )
    plot!(extrap_gamma, dose["altitude_km"], label = "μ_en flat extrapolated")
    plot!(ignore_gamma, dose["altitude_km"], linestyle = :dash, label = "High E gamma ingored")
    plot!(extrap_gamma .- ignore_gamma, dose["altitude_km"])
    hline!([11], color = :grey, linestyle = :dash)
    display(plot!())
end

function rep_explained_fraction_of_excess()
    rep_data = npzread("$(TOP_LEVEL)/data/figure_data/elfin_electron_doserate.npz")
    rep_dose = LinearInterpolator(rep_data["altitude_km"], rep_data["top_doserate_uGy/hr"], NoBoundaries())

    excess_data = npzread("$(TOP_LEVEL)/data/figure_data/conjunction_data.npz")
    altitude_bin_means = edges_to_means(excess_data["altitude_bin_edges"])
    doserate_bin_means = edges_to_means(excess_data["excess_dose_bin_edges"])


    explainable_datapoints = 0
    for i in eachindex(altitude_bin_means) 
        explainable_datapoints += sum(excess_data["excess_dose_histogram"][i, doserate_bin_means .≤ rep_dose(altitude_bin_means[i])])
    end

    total_datapoints = sum(excess_data["excess_dose_histogram"])

    @show 100 * (explainable_datapoints / total_datapoints)
end

function percentile(value, cdf_edges, cdf)
    normalized_cdf = cdf ./ cdf[end]
    idx = findlast(value .> cdf_edges[begin:end-1])
    if idx == nothing; return 0; end
    return 100 * normalized_cdf[idx]
end

function scratch_fig()
    start_times_to_save = [DateTime("2015-12-19T00:59:40"), DateTime("2019-09-14T10:40:20"), DateTime("2019-09-05T17:15:20"), DateTime("2017-10-14T21:08:10"), DateTime("2015-10-03T11:31:10")]
    vehicles_to_save = ["AFRC", "PN-2", "DC87", "DC-8", "G-V"]
    assets_to_save = ["FM1001", "FM6002", "FM5003", "FM1001", "FM2002"]

    i = 5
    data = get_armas_data(start_times_to_save[i], vehicles_to_save[i], assets_to_save[i])

    valid = data["data_valid"]
    time = data["time"][valid]
    nairas = data["NL2DrSi"][valid]
    armas = data["AL2DrSi"][valid]

    xticks = DateTime("2015-10-03T12:00:00"):Hour(2):DateTime("2015-10-04T1:00:00")
    plot(
        xlabel = "Oct. 03 2015 (UTC)",
        xlims = (min(time...), max(time...)),
        xticks = (xticks, Dates.format.(Time.(xticks), "HH:MM")),

        ylabel = "Doserate, μGy/hr",
        ylims = (0, 6),

        margin = 5mm,
        background_color_inside = :transparent,
        background_color_outside = :transparent,
        size = (2, .6) .* 350,
    )
    plot!(time, armas,
        label = "ARMAS (Data)",
        color = :black,
        linetype = :steppost
    )
    plot!(time, nairas,
        label = "NAIRAS (Model)",
        color = :blue,
        linewidth = 2
    )
    box_aspect!(1/5)
    display("image/png", plot!())
end

function move_ams_energies_from_results()
    source_dir = "/Users/luna/Research/geant4/Aviation_GLYPHS/_isotropic_beams"

    data = get_ams02_spectrum()
    ams_energies_keV = 1e6 .* data["proton_energy_bins_mean_GeV"]

    _, existing_energies = get_beams(source_dir, input_particle = "proton")

    closest_energies = similar(ams_energies_keV)
    for i in eachindex(ams_energies_keV)
        _, idx = findmin( abs.(ams_energies_keV[i] .- existing_energies) )
        closest_energies[i] = existing_energies[idx]
    end

    vline(log10.(ams_energies_keV))
    scatter!(log10.(closest_energies), ones(length(closest_energies)))
    display(plot!())

    destination_dir = results_dir
    for i in eachindex(closest_energies)
        E = @sprintf "%.1f" closest_energies[i]

        paths = glob("proton_input_$(E)keV_100000particles_*_spectra.csv", source_dir)
        for j in eachindex(paths)
            cp(paths[j], "$(destination_dir)/$(basename(paths[j]))", force = true)
            println("Copied $(basename(paths[j]))")
        end
    end
end

results_dir = "$(TOP_LEVEL)/data/GLYPHS"
#=
save_example_energy_histograms()
save_elfin_example_event()
save_stopping_power()
save_gcr_spectrum()
save_elfin_derived_doserates()
save_example_armas_data()

save_conjunction_data()
=#
#find_elfin_spectra()
#view_elfin_spectra("$(TOP_LEVEL)/results/percentile_98_elfin_spectra.csv")

#rep_explained_fraction_of_excess()
fraction_contributed_by_out_of_range_gammas()