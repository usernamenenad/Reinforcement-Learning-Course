function log_results(results::Dict{Int,Bool})
    successful = count(collect(values(results)))
    bar(["Unsuccessful"], [length(results) - successful], legend=false, color=:red, show=true)
    bar!(["Successful"], [successful], legend=false, color=:blue, show=true)
end