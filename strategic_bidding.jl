using JuMP, NLopt
using DataFrames, CSV
using SparseArrays
using LinearAlgebra
using Ipopt
using Random

include("nlp_utilities.jl")
include("nlp_utilities_test.jl")
include("opf.jl")

################################################
# Strategic bidding test
################################################

# Parameters
max_eval = 100
solver_lower = Ipopt.Optimizer
casename = "pglib_opf_case1354_pegase"# "pglib_opf_case300_ieee"
save_file = "results/strategic_bidding_nlopt_$(casename).csv"

# #### test
# solver_upper = :LD_MMA # :LD_MMA :LN_BOBYQA
# Random.seed!(1234)
# start_time = time()
# profit, num_evals, trace = test_bidding_nlopt(casename; percen_bidding_nodes=0.1, Δp=nothing, solver_lower=solver_lower, solver_upper=solver_upper, max_eval=2)
# end_time = time()
#### end test

# Experiments
seeds = collect(1:3)

experiements = Dict(
    :LD_MMA => [nothing], # 0.001
    :LN_BOBYQA => [0.0],
    :LD_CCSAQ => [nothing],
    :LD_SLSQP => [nothing],
    :LD_LBFGS => [nothing],
    :LD_TNEWTON_PRECOND_RESTART => [nothing],
    :LN_COBYLA => [0.0],
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
_experiments = [(string(solver_upper), string(solver_lower), string(Δp), seed) for (solver_upper, Δp_values) in experiements for Δp in Δp_values for seed in seeds]
if isfile(save_file)
    old_results = CSV.read(save_file, DataFrame)
    _experiments = setdiff(_experiments, [(string(row.solver_upper), string(row.solver_lower), string(row.Δp), row.seed) for row in eachrow(old_results)])
else
    open(save_file, "w") do f
        write(f, "solver_upper,solver_lower,Δp,seed,profit,num_evals,time\n")
    end
end

# Run experiments
for (_solver_upper, _, _Δp, seed) in _experiments
    solver_upper = Symbol(_solver_upper)
    Δp = _Δp == "nothing" ? nothing : parse(Float64, _Δp)
    Random.seed!(seed)
    start_time = time()
    profit, num_evals, trace = test_bidding_nlopt(casename; percen_bidding_nodes=0.1, Δp=Δp, solver_lower=solver_lower, solver_upper=solver_upper, max_eval=max_eval)
    end_time = time()
    # push!(results, (string(solver_upper), string(solver_lower), string(Δp), seed, profit, num_evals, end_time - start_time))
    open(save_file, "a") do f
        write(f, "$solver_upper,$solver_lower,$Δp,$seed,$profit,$num_evals,$(end_time - start_time)\n")
    end
end

# Save append results
if isempty(results)
    @info "No new results"
end
