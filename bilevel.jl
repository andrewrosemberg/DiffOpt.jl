using Distributed

# Add worker processes if needed
# addprocs()

@everywhere pkg_path = @__DIR__

@everywhere import Pkg

@everywhere Pkg.activate(pkg_path)

@everywhere Pkg.instantiate()

@everywhere using BilevelJuMP
@everywhere using Ipopt
@everywhere using QuadraticToBinary
@everywhere using Gurobi
@everywhere using LinearAlgebra
@everywhere using Combinatorics
@everywhere using StatsBase
@everywhere using Random

@everywhere function build_lower_level(num_nodes=10; jump_model=Model(), seed=1, qS=100)
    Random.seed!(seed)
    costs = rand(1:1:200, num_nodes)
    demands = rand(1:1:100, num_nodes)
    # circle network with num_nodes extra random lines
    comb = collect(combinations(1:num_nodes,2))
    num_comb = length(comb)
    rand_lines = comb[sample(1:num_comb, ceil(Int, num_nodes / 2), replace = false)]
    lines = vcat([(i, i+1) for i=1:num_nodes-1], [(num_nodes, 1)], rand_lines)
    num_lines = length(lines)
    line_limits = rand(1:1:100, num_lines)
    @variable(jump_model, 0 <= gS[i=1:num_nodes] <= 100)
    @variable(jump_model, 0 <= gR[i=1:num_nodes] <= 40)
    @variable(jump_model, 0 <= gD[i=1:num_nodes] <= 100)
    # line flows
    @variable(jump_model, 0 <= f[i=1:num_lines] <= line_limits[i])
    @objective(jump_model, Min, dot(costs, gR) + 1000 * sum(gD))
    @constraint(jump_model, gS .<= qS)
    # demand_equilibrium
    # @constraint(jump_model, demand_equilibrium, gS + sum(gR) + gD == 100)
    @constraint(jump_model, demand_equilibrium[i=1:num_nodes],
        gS[i] + gR[i] + gD[i] + sum(f[j] for j in 1:num_lines if lines[j][1] == i) - sum(f[j] for j in 1:num_lines if lines[j][2] == i) == demands[i]
    )
    return jump_model, demand_equilibrium, gS
end

@everywhere function build_bilevel(num_nodes=10; seed=1)
    model = BilevelModel()
    @variable(Upper(model), 0 <= qS[i=1:num_nodes] <= 100)
    _, demand_equilibrium, gS, = build_lower_level(num_nodes; jump_model=Lower(model), seed=seed, qS=qS)
    @variable(Upper(model), lambda[i=1:num_nodes], DualOf(demand_equilibrium[i]))
    @objective(Upper(model), Max, dot(lambda, gS))
    return model, qS, lambda
end

@everywhere function solve_nlp(model)
    BilevelJuMP.set_mode(model, BilevelJuMP.StrongDualityMode())
    set_optimizer(model, Ipopt.Optimizer)
    start_time = time()
        optimize!(model)
    end_time = time()
    return end_time - start_time
end

@everywhere function solve_bilevel_exact(model, lambda)
    set_optimizer(model,
        ()->QuadraticToBinary.Optimizer{Float64}(MOI.instantiate(optimizer_with_attributes(Gurobi.Optimizer, "MIPGap" => 0.1)))
    )
    BilevelJuMP.set_mode(model,
        BilevelJuMP.FortunyAmatMcCarlMode(dual_big_M = 1000)
    )
    set_lower_bound.(lambda, 0.0)
    set_upper_bound.(lambda, 1000.0)
    start_time = time()
        optimize!(model)
    end_time = time()
    return end_time - start_time
end

@everywhere function calculate_profit(evaluator_lower_level, demand_equilibrium, gS, _qS, qS)
    fix.(_qS, value.(qS))
    # @constraint(evaluator_lower_level, _qS .== value.(qS))
    optimize!(evaluator_lower_level)
    return dot(value.(gS), dual.(demand_equilibrium))
end

@everywhere function compare_methods(num_nodes=10; seed=1)
    # try
    bilevel_model, qS, lambda = build_bilevel(num_nodes; seed=seed)
    evaluator_lower_level = JuMP.Model(Gurobi.Optimizer)
    @variable(evaluator_lower_level, _qS[i=1:num_nodes])
    evaluator_lower_level, demand_equilibrium, gS = build_lower_level(num_nodes; seed=seed, jump_model=evaluator_lower_level, qS=_qS)
    nlp_time = solve_nlp(bilevel_model)
    opf_time = time()
    nlp_profit = calculate_profit(evaluator_lower_level, demand_equilibrium, gS, _qS, qS)
    opf_time = time() - opf_time
    exact_time = solve_bilevel_exact(bilevel_model, lambda)
    # exact_profit = calculate_profit(evaluator_lower_level, demand_equilibrium, gS, _qS, qS)
    return nlp_time / opf_time, exact_time / opf_time
    # catch e
    #     return NaN, NaN
    # end
end

nodes = 7:5:57
results = pmap(x->compare_methods(x), nodes)

# success nodes
_results = [(i, (r[1], r[2])) for (i,r) in enumerate(results) if !isnan(r[1]) && !isnan(r[2])]
nodes = [nodes[r[1]] for r in _results]
results = [r[2] for r in _results]

# Save results
using CSV, DataFrames
df = DataFrame(nodes=nodes, nlp=[r[1] for r in results], exact=[r[2] for r in results])
if isfile(joinpath(@__DIR__, "results", "complexity_mpec.csv"))
    df_old = CSV.read(joinpath(@__DIR__, "results", "complexity_mpec.csv"), DataFrame)
    df = vcat(df_old, df)
end
CSV.write(joinpath(@__DIR__, "results", "complexity_mpec.csv"), df)

using Plots
# log y axis scale
plt = Plots.plot(df.nodes, df.nlp, label="NLP", xlabel="Number of nodes", ylabel="Time (x OPF)", yscale=:log10)
Plots.plot!(plt, df.nodes, df.exact, label="Exact", xlabel="Number of nodes", ylabel="Time (x OPF)", yscale=:log10)
Plots.savefig(plt, joinpath(@__DIR__, "results", "complexity_mpec.pdf"))



