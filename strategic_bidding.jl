using JuMP
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

using JuMP, NLopt
using DataFrames, CSV

solver_upper = :LN_BOBYQA # :LD_MMA
max_eval = 100
solver_lower = Ipopt.Optimizer
casename = "pglib_opf_case300_ieee"
Random.seed!(1234)

start_time = time()
profit, num_evals, trace = test_bidding_nlopt(casename; percen_bidding_nodes=0.1, Δp=0.0, solver_lower=solver_lower, solver_upper=solver_upper, max_eval=max_eval)
end_time = time()

solvers_upper = [:LD_MMA, :LN_BOBYQA]
Δps = [0.0, 0.001, nothing]
seeds = [1234, 1235, 1236]

results = DataFrame(
    solver_upper = String[],
    Δp = Union{Nothing, Float64}[],
    seed = Int[],
    profit = Float64[],
    num_evals = Int[],
    time = Float64[]
)

for solver_upper in solvers_upper
    for Δp in Δps
        for seed in seeds
            Random.seed!(seed)
            start_time = time()
            profit, num_evals, trace = test_bidding_nlopt(casename; percen_bidding_nodes=0.1, Δp=Δp, solver_lower=solver_lower, solver_upper=solver_upper, max_eval=max_eval)
            end_time = time()
            push!(results, (solver_upper, Δp, seed, profit, num_evals, end_time - start_time))
        end
    end
end

using CSV
save_file = "results/strategic_bidding_nlopt_$(casename).csv"

CSV.write(save_file, results)