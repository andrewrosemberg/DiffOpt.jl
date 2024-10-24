using Distributed

# Add worker processes if needed
# addprocs()

@everywhere pkg_path = @__DIR__

@everywhere import Pkg

@everywhere Pkg.activate(pkg_path)

@everywhere Pkg.instantiate()

@everywhere using JuMP, NLopt
@everywhere using DataFrames, CSV
@everywhere using SparseArrays
@everywhere using LinearAlgebra
@everywhere using Ipopt
@everywhere using Random
@everywhere using Logging

@everywhere include("nlp_utilities.jl")
@everywhere include("nlp_utilities_test.jl")
@everywhere include("opf.jl")

################################################
# Strategic bidding test
################################################

# Parameters
@everywhere max_eval = 100
@everywhere solver_lower_name = "Ipopt"
@everywhere casename = "pglib_opf_case1354_pegase" # "pglib_opf_case300_ieee" "pglib_opf_case1354_pegase" "pglib_opf_case2869_pegase"
@everywhere save_file_name = "results/strategic_bidding_nlopt_$(casename)"
@everywhere save_file = save_file_name * ".csv"

@everywhere data = make_basic_network(pglib(casename))

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
    # :LD_MMA => [nothing],
    # :LN_BOBYQA => [0.0],
    # :LD_CCSAQ => [nothing],
    # :LD_SLSQP => [nothing],
    # :LD_LBFGS => [nothing],
    # :LD_TNEWTON_PRECOND_RESTART => [nothing],
    # :LN_COBYLA => [0.0],
    :LN_NELDERMEAD => [0.0],
    :LN_NEWUOA_BOUND => [0.0]
)

@everywhere res = Dict(
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

# Prepare the list of experiments
_experiments = [
    (string(solver_upper), solver_lower_name, string(Δp), seed)
    for (solver_upper, Δp_values) in experiements
    for Δp in Δp_values
    for seed in seeds
]

# Check already executed experiments
if isfile(save_file)
    old_results = CSV.read(save_file, DataFrame)
    _experiments = setdiff(_experiments, [
        (string(row.solver_upper), string(row.solver_lower), string(row.Δp), row.seed)
        for row in eachrow(old_results)
    ])
else
    open(save_file, "w") do f
        write(f, "solver_upper,solver_lower,Δp,seed,profit,market_share,num_evals,time,status\n")
    end
end

@everywhere function run_experiment(_solver_upper, _Δp, seed, id)
    _data = deepcopy(data)
    solver_upper = Symbol(_solver_upper)
    Δp = _Δp == "nothing" ? nothing : parse(Float64, _Δp)
    try
        Random.seed!(seed)
        start_time = time()
        profit, num_evals, trace, market_share, ret = test_bidding_nlopt(
            _data; percen_bidding_nodes=0.1, Δp=Δp, solver_upper=solver_upper, max_eval=max_eval
        )
        end_time = time()
        ret = res[ret]
        if ret < 0
            @warn "Solver $(solver_upper) failed with seed $(seed)"
            return nothing
        else
            df = DataFrame(
                solver_upper = [string(solver_upper)],
                solver_lower = [solver_lower_name],
                Δp = [string(Δp)],
                seed = [seed],
                profit = [profit],
                market_share = [market_share],
                num_evals = [num_evals],
                time = [end_time - start_time],
                status = [ret]
            )
            return df
        end
    catch e
        @warn "Solver $(solver_upper) failed with seed $(seed)" e
        return nothing
    end
end

# Run experiments using pmap
experiments_list = [
    (_solver_upper, _Δp, seed, id)
    for (id, (_solver_upper, _, _Δp, seed)) in enumerate(_experiments)
]

results = pmap(
    experiment -> run_experiment(experiment...),
    experiments_list
)

# Filter out failed experiments
results = filter(!isnothing, results)

# Combine results into a DataFrame
if !isempty(results)
    all_results = vcat(results...)
    # Append to existing results if any
    if isfile(save_file)
        old_results = CSV.read(save_file, DataFrame)
        combined_results = vcat(old_results, all_results)
    else
        combined_results = all_results
    end
    # Save combined results
    CSV.write(save_file, combined_results)
else
    @info "No new results"
end

# Proceed with plotting or further analysis as needed


# Plot results: One scatter point per solver_upper
# - Each solver_upper should be an interval with the mean and std of the results per seed
# - x-axis: market_share | y-axis: (profit / max_profit) * 100
# - color: one per solver_upper
# - shape: one per Δp

using Plots
using Statistics

results = CSV.read(save_file, DataFrame)

ignore_solver_upper = [:LD_LBFGS]

results = results[[!(Symbol(sv_up) ∈ ignore_solver_upper) for sv_up in results.solver_upper], :]

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