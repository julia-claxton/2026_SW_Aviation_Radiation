using Statistics, LinearAlgebra          # Core math
using BenchmarkTools, Profile, TickTock  # Debugging
using NPZ, DelimitedFiles                # File interactions
using Dates
include("./General_Functions.jl")

const TOP_LEVEL = dirname(@__DIR__)

function get_ams02_coverage()
    data = readdlm("$(TOP_LEVEL)/data/AMS-02/p_AMS_PRL2021_daily.csv", ',', skipstart = 1)
    return unique(Date.(data[:,1], dateformat"yyyy-mm-dd"))
end

function get_ams02_spectrum(; date = nothing)
    proton_path = "$(TOP_LEVEL)/data/AMS-02/p_AMS_PRL2021_daily.csv"
    alpha_path = "$(TOP_LEVEL)/data/AMS-02/He_AMS_PRL2022_daily.csv"

    # Select random date if user doesn't specify date
    data = readdlm(proton_path, ',', skipstart = 1)
    dates = Date.(data[:,1], dateformat"yyyy-mm-dd")
    if date == nothing; date = rand(dates); end

    results = Dict{String, Vector{Float64}}()
    results = merge(results, read_ams02_file(proton_path, date, 1, 1, "proton"))
    results = merge(results, read_ams02_file(alpha_path, date, 2, 4, "alpha"))

    return results
end

function read_ams02_file(path, date, Z, A, particle_name)
    # Z = Charge number. Units: qₑ (number of elementary charges)
    # A = Mass number.   Units: mₚ (number of proton masses)

    data = readdlm(path, ',', skipstart = 1)
    dates = Date.(data[:,1], dateformat"yyyy-mm-dd")
    slice = dates .== date

    if sum(slice) == 0
        error("Date $(date) not found in AMS-02 data.")
    end

    # Read data from file
    rigidity_min_GV = float.(data[slice, 2])      # Units: GV
    rigidity_max_GV = float.(data[slice, 3])      # Units: GV    
    flux = float.(data[slice, 4])          # Units: # / (m2 sr s GV)
    statistical_error = float.(data[slice, 5])    # Units: # / (m2 sr s GV)
    time_dependent_error = float.(data[slice, 6]) # Units: # / (m2 sr s GV)
    total_error = float.(data[slice, 7])          # Units: # / (m2 sr s GV)
    
    # Get bin means
    rigidity_mean_GV = edges_to_means([rigidity_min_GV..., rigidity_max_GV[end]])

    # Convert GV to GeV
    energy_bins_min_GeV  = ridigity_GV_to_energy_GeV.(Z, A, rigidity_min_GV)
    energy_bins_mean_GeV = ridigity_GV_to_energy_GeV.(Z, A, rigidity_mean_GV)
    energy_bins_max_GeV  = ridigity_GV_to_energy_GeV.(Z, A, rigidity_max_GV)
    
    ΔR_GV  = rigidity_max_GV .- rigidity_min_GV
    ΔE_GeV = energy_bins_max_GeV .- energy_bins_min_GeV

    flux .*= (ΔR_GV ./ ΔE_GeV)
    total_error .*= (ΔR_GV ./ ΔE_GeV)

    # Construct data structure and return
    return Dict(
        "$(particle_name)_energy_bins_min_GeV" => energy_bins_min_GeV,   # Units: GeV
        "$(particle_name)_energy_bins_mean_GeV" => energy_bins_mean_GeV, # Units: GeV
        "$(particle_name)_energy_bins_max_GeV" => energy_bins_max_GeV,   # Units: GeV
        "$(particle_name)_flux" => flux,                                 # Units: # / (m2 sr s GeV)
        "$(particle_name)_total_error" => total_error                    # Units: # / (m2 sr s GeV)
    )
end

function ridigity_GV_to_energy_GeV(Z, A, R)
    # Z = Charge number. Units: qₑ (number of elementary charges)
    # A = Mass number.   Units: mₚ (number of proton masses)
    # R = Rigidity.      Units: GeV
    
    # Constants
    mp = 1.67e-27 # Proton mass. Units: kg
    qe = 1.602e-19 # Elementary charge. Units: C
    c_light = 3e8 # Speed of light. Units: m/s

    # Intermediate calculations
    R = copy(R) * 1e9 # Rigidity. Units: V
    E0 = (mp * c_light^2) / 1.6e-19 # Units: eV / nucleon

    # Solve quadratic for kinetic energy per nucleon (T)
    a = 1
    b = 2*E0
    c = -(R * Z / A )^2
    T = (-b + sqrt(b^2 - 4*a*c))/(2a) # Units: eV / nucleon

    # Convert to total kinetic energy in GeV and return
    KE = T * A # Units: eV
    return KE/1e9 # Units: GeV
end

