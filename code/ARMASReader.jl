using Statistics, LinearAlgebra          # Core math
using BenchmarkTools, Profile, TickTock  # Debugging
using NPZ, DelimitedFiles                # File interactions
using Dates
using Glob
include("./General_Functions.jl")

const TOP_LEVEL = dirname(@__DIR__)

function get_armas_L_MLT_coverage()
    coverage_data = get_armas_coverage()
    N = length(coverage_data["start_times"])

    MLT_bin_edges = 0:.5:24
    L_bin_edges = 0:.5:20
    coverage = zeros(length(MLT_bin_edges)-1, length(L_bin_edges)-1)

    for i in 1:N
        print_progress_bar(i/N)

        data = get_armas_data(coverage_data, i)
        if data["VehID"][1] == "Lunar"; continue; end
        valid = data["data_valid"]

        l = data["Lshell"][valid]
        mlt = data["mlt"][valid]
        mlt = floor.(mlt./100) + ((mlt.%60)./60)

        coverage .+= exact_2dhistogram(mlt, l, MLT_bin_edges, L_bin_edges)
    end
    return MLT_bin_edges, L_bin_edges, coverage
end

function get_armas_coverage()
    armas_dir = "$(TOP_LEVEL)/data/ARMAS"

    # Make preprocessed dir if needed
    if isdir("$(armas_dir)/preprocessed") == false
        mkdir("$(armas_dir)/preprocessed")
    end

    # Return preprocessed file if it already exists
    coverage_path = "$(armas_dir)/preprocessed/armas_coverage.csv"
    if isfile(coverage_path)
        coverage_data = readdlm(coverage_path, ',', skipstart = 1)
        return Dict(
            "start_times" => DateTime.(coverage_data[:, 1]),
            "vehicles" => string.(coverage_data[:, 2]),
            "assets" => string.(coverage_data[:, 3])
        )
    end

    # Do preprocessing
    println("Collecting ARMAS coverage...")
    paths = readdir(armas_dir)
    paths = paths[endswith.(paths, ".txt")] # Filter out non-data files

    dates = Any["date"] # Prefill with header values
    vehicles = Any["vehicle_id"]
    assets = Any["asset"]
    for i in eachindex(paths)
        print_progress_bar(i/length(paths))
        data = armas_read_file("$(armas_dir)/$(paths[i])")

        push!(dates, data["time"][begin] )
        push!(vehicles, data["VehID"][begin])
        push!(assets, data["Asset"][begin])
    end
    writedlm(coverage_path, hcat(dates, vehicles, assets), ',')
    
    # Return. RECURSION, watch out!
    return get_armas_coverage()
end

function get_armas_data(coverage_data::Dict, index::Int)
    return get_armas_data(coverage_data["start_times"][index], coverage_data["vehicles"][index], coverage_data["assets"][index])
end

function get_armas_data(start_time::DateTime, vehicle::String, asset::String)
    armas_dir = "$(TOP_LEVEL)/data/ARMAS"
    if isdir("$(armas_dir)/preprocessed") == false
        preprocess_armas()
    end

    data_filename = "$(Dates.format(start_time, "yyyymmddHHMMSS"))_$(vehicle)_$(asset).txt"
    data_path = "$(armas_dir)/preprocessed/$(data_filename)"

    if isfile(data_path) == false
        error("File $(data_path) not found")
    end

    return armas_read_file(data_path)
end

function preprocess_armas()
    # The naming scheme in the ARMAS raw data is too ambiguous, so copy them to a new folder with unique names
    armas_dir = "$(TOP_LEVEL)/data/ARMAS"

    # Make preprocessed dir if needed
    if isdir("$(armas_dir)/preprocessed") == false
        mkdir("$(armas_dir)/preprocessed")
    end

    # Do preprocessing
    println("Renaming ARMAS data...")
    paths = readdir(armas_dir)
    paths = paths[endswith.(paths, ".txt")] # Filter out non-data files

    for i in eachindex(paths)
        print_progress_bar(i/length(paths))
        data = armas_read_file("$(armas_dir)/$(paths[i])")
        start_time = Dates.format(data["time"][begin], "yyyymmddHHMMSS")
        vehicle = data["VehID"][begin]
        asset = data["Asset"][begin]

        destination_filename = "$(start_time)_$(vehicle)_$(asset).txt"
        destination_path = "$(armas_dir)/preprocessed/$(destination_filename)"

        # Overwrite existing records
        if isfile(destination_path)
            rm(destination_path)
        end

        symlink("$(armas_dir)/$(paths[i])", destination_path)
    end
end

function armas_read_file(path)
    # Read in data to a string array
    if islink(path); path = readlink(path); end
    file_contents = readlines(path)

    # SEP event flag
    sep_idx = findfirst([contains(file_contents[i], ":Flight_SEP: SEP for this flight is") for i in eachindex(file_contents)])
    sep = parse(Int, file_contents[sep_idx][end])

    # Get headers
    header_idx = findfirst([startswith(el, "#year mo") for el in file_contents ])
    headers = armas_read_row(
        replace(file_contents[header_idx], "#" => " ", "other comments" => "other_comments")
    )

    # Get data
    rows = [el for el in file_contents if el[1] ∉ [':', '#']]
    data = vcat([permutedims(armas_read_row.(rows[i])) for i in eachindex(rows)]...)

    # Parse data to dict
    result = Dict()
    result["sep"] = sep
    result["time"] = [DateTime(parse.(Int, data[i, 1:6])...) for i in 1:length(rows)]
    result["local_time"] = Time.(replace.(data[:, 7], "0-" => "00"), dateformat"HH:MM")
    result["mlt"] = parse.(Int, replace.(data[:, 8], ":" => ""))
    for i in 9:55
        result[headers[i]] = parse.(Float64, data[:, i])
    end
    for i in 56:58
        result[headers[i]] = data[:, i]
    end
    # Add 'other comments' field if present
    result["other_comments"] = fill("", size(data)[1])
    if size(data)[2] > 59
        result["other_comments"] = data[:, 59:end-1]
    end
    result["data_valid"] = data[:, end] .== "valid_data"
    
    return result
end

function armas_read_row(row::String)
    is_data = [el ≠ ' ' for el in row]
    n_cols = sum(diff(is_data) .== 1)
    results = repeat([""], n_cols)

    result_idx = 0
    for i in eachindex(row)
        # Skip if not data
        if is_data[i] == false; continue; end

        # If we are at data, increment result index if its the first character of a new column
        if is_data[i-1] == false
            result_idx += 1
        end

        # Put data into results vector
        results[result_idx] = string(results[result_idx], row[i])
    end
    return results
end

function example_armas()
    return armas_read_file(rand(readdir("$(TOP_LEVEL)/data/ARMAS", join = true)))
end