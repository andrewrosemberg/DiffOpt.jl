using DataFrames, Statistics, CSV

function compute_improvements(df::DataFrame)
    # Get all unique seeds in the dataframe
    seeds = unique(df.seed)
    
    best_improvements = Float64[]
    overall_improvements = Float64[]
    
    for s in seeds
        # Filter data for the current seed
        df_seed = df[df.seed .== s, :]
        
        # Select rows for LD and LN methods.
        # We assume that the 'solver_upper' column indicates "LD" or "LN" at the beginning.
        ld = df_seed[startswith.(df_seed.solver_upper, "LD"), :]
        ln = df_seed[startswith.(df_seed.solver_upper, "LN"), :]
        
        # Skip this seed if one of the groups is missing
        if nrow(ld) == 0 || nrow(ln) == 0
            continue
        end
        
        # For the best improvement: select the row with the best (highest) profit from each group.
        ld_best = ld[findmax(ld.profit)[2], :]
        ln_best = ln[findmax(ln.profit)[2], :]
        
        # Compute percentage improvement: how much lower LD's evaluations are compared to LN's.
        best_impr = ((ln_best.num_evals - ld_best.num_evals) / ln_best.num_evals) * 100
        push!(best_improvements, best_impr)
        
        # For overall improvement: compute average num_evals for LD and LN.
        ld_avg = mean(ld.num_evals)
        ln_avg = mean(ln.num_evals)
        overall_impr = ((ln_avg - ld_avg) / ln_avg) * 100
        push!(overall_improvements, overall_impr)
    end
    
    # Compute overall averages and standard deviations across seeds.
    best_avg = mean(best_improvements)
    best_std = std(best_improvements)
    overall_avg = mean(overall_improvements)
    overall_std = std(overall_improvements)
    
    # Return the results as a one-row DataFrame.
    return DataFrame(
        "Best Improvement AVG" => best_avg,
        "Best Improvement STD" => best_std,
        "Overall Improvement AVG" => overall_avg,
        "Overall Improvement STD" => overall_std,
    )
end

# Example usage
prefix = "strategic_bidding_nlopt_"
case_names = ["pglib_opf_case300_ieee" "pglib_opf_case1354_pegase" "pglib_opf_case2869_pegase" "pglib_opf_case2000_goc" "pglib_opf_case2868_rte.m"]
df = DataFrame()
for case in case_names
    file = "results/$(prefix)$(case).csv"
    df_case = CSV.read(file, DataFrame)
    df_results = compute_improvements(df_case)
    df_results.case = [case]
    append!(df, df_results)
end

CSV.write("results/time_summary.csv", df)