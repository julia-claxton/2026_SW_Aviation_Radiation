using Statistics, LinearAlgebra          # Core math
using BenchmarkTools, Profile, TickTock  # Debugging
using Glob
using Printf
using NPZ
using DelimitedFiles
include("./General_Functions.jl")

const TOP_LEVEL = dirname(@__DIR__)

function get_beams(
    # Return the input particle, input energy, and input pitch angle for all Geant4 beams contained in results_dir
    results_dir::String # Directory to search for Geant4 beam data in
    ;
    input_particle = "all"
    )
    available_files = glob("*_input_*keV_*particles_electron_spectra.csv", results_dir) # Arbitarily choose electron spectra to not double-count multiple result files from the same beam

    # Get particle type
    matches = match.(Regex("(.*?)_input"), basename.(available_files))
    particle_type = [string(el.captures[1]) for el in matches]

    # Get energy
    matches = match.(Regex("input_(.*?)keV_"), basename.(available_files))
    energy_keV = parse.(Float64, [el.captures[1] for el in matches])

    # Filter by input particle if needed
    if input_particle ≠ "all"
        mask = particle_type .== input_particle
        particle_type = particle_type[mask]
        energy_keV = energy_keV[mask]
    end

    # Return
    sortvec = sortperm(energy_keV)
    return particle_type[sortvec], energy_keV[sortvec]
end

function preprocess_spectra(results_dir, input_particle, input_energy_keV)
    # Load data
    energy_string = @sprintf "%.1f" input_energy_keV
    paths = glob("$(input_particle)_input_$(energy_string)keV_*.csv", results_dir)

    # Try rounding if there are no matching paths
    if length(paths) == 0
        energy_string = @sprintf "%.0i" input_energy_keV
        paths = glob("$(input_particle)_input_$(energy_string)keV_*.csv", results_dir)
    end

    # Error if we can't find raw csv
    if length(paths) == 0
        error("No files matching pattern \"$(input_particle)_input_$(energy_string)keV_*.csv\" found in $(results_dir)")
    end 

    # Parse beam information
    matches = match.(Regex("keV_(.*?)particles"), basename.(paths))
    n_particles = [parse(Int64, el[1]) for el in matches]
    highest_n_particles = max(n_particles...)
    
    # Get spectra
    results = Dict{String, Any}()
    results["n_particles"] = highest_n_particles
    results["input_energy_keV"] = input_energy_keV

    for particle in ["electron", "proton", "gamma", "alpha"]
        path = glob("$(input_particle)_input_$(energy_string)keV_$(highest_n_particles)particles_$(particle)_spectra.csv", results_dir)[1]
        data = readdlm(path, ',', skipstart = 0)
        counts = float.(data[2:end, 2:end])

        # Parse altitude labels
        altitude_km = float.(data[2:end, 1])

        # Only pick half-km altitudes due to combing issue at geometry boundaries
        alt_idxs = (altitude_km .% 1) .≈ 0.45

        # Parse energy labels
        energy_labels = data[1, 2:end]
        energy_bins_min_keV = zeros(length(energy_labels))
        energy_bins_max_keV = zeros(length(energy_labels))

        for i in eachindex(energy_labels)
            label = energy_labels[i]

            matches = match(Regex("(.*?)keV-(.*?)keV"), label) # Matches for the pattern containing energy limits
            energy_bins_min_keV[i] = parse(Float64, matches[1])
            energy_bins_max_keV[i] = parse(Float64, matches[2])
        end

        # Ensure labels match between spectra
        if "altitude_km" in keys(results)
            @assert altitude_km[alt_idxs] == results["altitude_km"]
        end
        if "energy_bins_min_keV" in keys(results)
            @assert energy_bins_min_keV == results["energy_bins_min_keV"]
        end
        if "energy_bins_max_keV" in keys(results)
            @assert energy_bins_max_keV == results["energy_bins_max_keV"]
        end

        # Assign labels
        results["altitude_km"] = altitude_km[alt_idxs]
        results["energy_bins_min_keV"] = energy_bins_min_keV
        results["energy_bins_max_keV"] = energy_bins_max_keV
        results["energy_bins_mean_keV"] = edges_to_means([energy_bins_min_keV..., energy_bins_min_keV[end]])

        # Assign counts
        results["$(particle)_counts"] = counts[alt_idxs, :]
    end

    # Write result
    npzwrite("$(results_dir)/preprocessed/$(input_particle)_input_$(energy_string)keV_$(highest_n_particles)particles_spectra.npz", results)
end

function get_spectra(results_dir, input_particle, input_energy_keV)
    # Load data
    processed_dir = "$(results_dir)/preprocessed"
    if isdir(processed_dir) ≠ true; mkdir(processed_dir); end
    energy_string = @sprintf "%.1f" input_energy_keV
    paths = glob("$(input_particle)_input_$(energy_string)keV_*.npz", processed_dir)

    # Try rounding if there are no matching paths
    if length(paths) == 0
        energy_string = @sprintf "%.0i" input_energy_keV
        paths = glob("$(input_particle)_input_$(energy_string)keV_*.npz", processed_dir)
    end

    # Try to preprocess if we can't find a file
    if length(paths) == 0
        preprocess_spectra(results_dir, input_particle, input_energy_keV)

        # RECURSION!!!! This shouldn't loop forever, but everyone thinks that so be careful. 
        return get_spectra(results_dir, input_particle, input_energy_keV) 
    end     
    
    # Get matching file with highest number of particles
    matches = match.(Regex("keV_(.*?)particles"), basename.(paths))
    n_particles = [parse(Int64, el[1]) for el in matches]
    highest_n_particles = max(n_particles...)

    # Get data and append the input particle to the dict because Julia NPZ can't handle strings in dicts :/
    data = npzread("$(results_dir)/preprocessed/$(input_particle)_input_$(energy_string)keV_$(highest_n_particles)particles_spectra.npz")
    data["input_particle"] = input_particle

    return data
end