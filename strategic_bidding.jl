using JuMP, NLopt
using DataFrames, CSV
using SparseArrays
using LinearAlgebra
using Ipopt
using Random
using Logging

include("nlp_utilities.jl")
include("nlp_utilities_test.jl")
include("opf.jl")

################################################
# Strategic bidding test
################################################

# Parameters
max_eval = 100
solver_lower, solver_lower_name = optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0), "Ipopt"
casename = "pglib_opf_case2869_pegase" # "pglib_opf_case300_ieee" "pglib_opf_case1354_pegase" "pglib_opf_case2869_pegase"
save_file = "results/strategic_bidding_nlopt_$(casename).csv"

# #### test Range Evaluation
# Random.seed!(1)
# data = build_bidding_opf_model(casename)
# # offer = rand(length(data["bidding_lmps"]))
# # offer = zeros(length(data["bidding_lmps"])); offer[1] = 1.0; offer[2] = 1.0; offer[3] = 1.0; offer[4] = 1.0
# offer = ones(length(data["bidding_lmps"]))
# profits = []
# start_time = time()
# for val = 0.0:0.1:12
#     set_parameter_value.(data["bid"], val * offer)
#     JuMP.optimize!(data["model"])
#     push!(profits, dot(-dual.(data["bidding_lmps"]), value.(data["bidding_generators_dispatch"])))
# end
# end_time = time()
#### end test

# #### test SB
# solver_upper = :LD_MMA # :LD_MMA :LN_BOBYQA
# Random.seed!(1234)
# start_time = time()
# profit, num_evals, trace, market_share, ret = test_bidding_nlopt(casename; percen_bidding_nodes=0.1, Δp=nothing, solver_lower=solver_lower, solver_upper=solver_upper, max_eval=2)
# end_time = time()
#### end test

# Experiments
seeds = collect(1:10)

experiements = Dict(
    :LD_MMA => [nothing], # 0.001
    :LN_BOBYQA => [0.0],
    :LD_CCSAQ => [nothing],
    :LD_SLSQP => [nothing],
    :LD_LBFGS => [nothing],
    :LD_TNEWTON_PRECOND_RESTART => [nothing],
    :LN_COBYLA => [0.0],
)

res = Dict(
    :FORCED_STOP => -5,
    :ROUNDOFF_LIMITED => -4,
    :OUT_OF_MEMORY => -3,
    :INVALID_ARGS => -2,
    :FAILURE => -1,
    :SUCCESS => 1,
    :STOPVAL_REACHED => 2,
    :FTOL_REACHED => 3,
    :XTOL_REACHED => 4,
    :MAXEVAL_REACHED => 5,
    :MAXTIME_REACHED => 6
)

# results = DataFrame(
#     solver_upper = String[],
#     solver_lower = String[],
#     Δp = String[],
#     seed = Int[],
#     profit = Float64[],
#     num_evals = Int[],
#     time = Float64[]
# )

# Check already executed experiments
_experiments = [(string(solver_upper), solver_lower_name, string(Δp), seed) for (solver_upper, Δp_values) in experiements for Δp in Δp_values for seed in seeds]
if isfile(save_file)
    old_results = CSV.read(save_file, DataFrame)
    _experiments = setdiff(_experiments, [(string(row.solver_upper), string(row.solver_lower), string(row.Δp), row.seed) for row in eachrow(old_results)])
else
    # open(save_file, "w") do f
    #     write(f, "solver_upper,solver_lower,Δp,seed,profit,market_share,num_evals,time,status\n")
    # end
end

for thread_id in 1:Threads.nthreads()
    open(save_file * "_$thread_id", "w") do f
        write(f, "solver_upper,solver_lower,Δp,seed,profit,market_share,num_evals,time,status\n")
    end
end

# Run experiments
# for (_solver_upper, _, _Δp, seed) in _experiments
function run_experiment(_solver_upper, _Δp, seed, id)
    solver_upper = Symbol(_solver_upper)
    Δp = _Δp == "nothing" ? nothing : parse(Float64, _Δp)
    try
        Random.seed!(seed)
        start_time = time()
        profit, num_evals, trace, market_share, ret = test_bidding_nlopt(casename; percen_bidding_nodes=0.1, Δp=Δp, solver_lower=solver_lower, solver_upper=solver_upper, max_eval=max_eval)
        end_time = time()
        # push!(results, (string(solver_upper), solver_lower_name, string(Δp), seed, profit, num_evals, end_time - start_time))
        ret = res[ret]
        if ret < 0
            @warn "Solver $(solver_upper) failed with seed $(seed)"
            continue
        else
            open(save_file * "_$thread_id", "a") do f
                write(f, "$solver_upper,$solver_lower_name,$Δp,$seed,$profit,$market_share,$num_evals,$(end_time - start_time),$ret\n")
            end
        end
    catch e
        @warn "Solver $(solver_upper) failed with seed $(seed)"
        continue
    end
end

# Run experiments on multiple threads
Threads.@threads for (_solver_upper, _solver_lower, _Δp, seed) in _experiments
    run_experiment(_solver_upper, _Δp, seed)
end

# Merge results
for thread_id in 1:Threads.nthreads()
    open(save_file * "_$thread_id", "r") do f
        lines = readlines(f)
        open(save_file, "a") do f
            for line in lines
                write(f, line)
            end
        end
    end
end

# Save append results
if isempty(_experiments)
    @info "No new results"
end

# Plot results: One scatter point per solver_upper
# - Each solver_upper should be an interval with the mean and std of the results per seed
# - x-axis: market_share | y-axis: (profit / max_profit) * 100
# - color: one per solver_upper
# - shape: one per Δp

using Plots
using Statistics

results = CSV.read(save_file, DataFrame)

maximum_per_seed = [maximum(results.profit[results.seed .== seed]) for seed in seeds]

results.gap = (maximum_per_seed[results.seed] - results.profit) * 100 ./ maximum_per_seed[results.seed]

# use combined groupby
results_d = combine(groupby(results, [:solver_upper, :Δp]), :gap => mean, :market_share => mean, :gap => std, :num_evals => mean, :time => mean)

# plot
markers = []
for Δp in results_d.Δp
    if Δp == "nothing"
        push!(markers, :circle)
    else
        push!(markers, :diamond)
    end
end
plt = scatter(results_d.market_share_mean, results_d.gap_mean, group=results_d.solver_upper, yerr=results_d.gap_std/2, 
    xlabel="Bid Volume (% of Market Share)", ylabel="Optimality Gap (%)", legend=:outertopright, marker=markers
)

# save
savefig(plt, "results/strategic_bidding_nlopt_$(casename).pdf")