#!/usr/bin/env julia
###### AC-OPF using JuMP ######
#
# implementation reference: https://github.com/lanl-ansi/PowerModelsAnnex.jl/blob/master/src/model/ac-opf.jl
# only the built-in AD library is supported
#

using PowerModels
using PGLib
import Ipopt
import JuMP
using LinearAlgebra
using ForwardDiff

function build_gen(gen_bus, id, pmax, qmax)
    return Dict{String, Any}(
        "pg"         => 0.5,
        "model"      => 2,
        "shutdown"   => 0.0,
        "startup"    => 0.0,
        "qg"         => 0.0,
        "gen_bus"    => gen_bus,
        "pmax"       => pmax,
        "vg"         => 1.0,
        "mbase"      => 100.0,
        "source_id"  => Any["gen", id],
        "index"      => id,
        "cost"       => [0.0, 0.0, 0.0],
        "qmax"       => qmax,
        "gen_status" => 1,
        "qmin"       => - qmax,
        "pmin"       => 0.0,
        "ncost"      => 3,
    )
end

function build_opf_model(data; add_param_load=false, solver=Ipopt.Optimizer)
    # create ref
    PowerModels.standardize_cost_terms!(data, order=2)
    PowerModels.calc_thermal_limits!(data)
    ref = PowerModels.build_ref(data)[:it][:pm][:nw][0]

    # create model
    model = JuMP.Model(solver)

    JuMP.@variable(model, va[i in keys(ref[:bus])])
    JuMP.@variable(model, ref[:bus][i]["vmin"] <= vm[i in keys(ref[:bus])] <= ref[:bus][i]["vmax"], start=1.0)
    for i in keys(ref[:bus])
        JuMP.set_name(va[i], "va_$i")
        JuMP.set_name(vm[i], "vm_$i")
    end
    JuMP.@variable(model, ref[:gen][i]["pmin"] <= pg[i in keys(ref[:gen])] <= ref[:gen][i]["pmax"])
    JuMP.@variable(model, ref[:gen][i]["qmin"] <= qg[i in keys(ref[:gen])] <= ref[:gen][i]["qmax"])

    for i in keys(ref[:gen])
        JuMP.set_name(pg[i], "pg_$i")
        JuMP.set_name(qg[i], "qg_$i")
    end
    JuMP.@variable(model, -ref[:branch][l]["rate_a"] <= p[(l,i,j) in ref[:arcs]] <= ref[:branch][l]["rate_a"])
    JuMP.@variable(model, -ref[:branch][l]["rate_a"] <= q[(l,i,j) in ref[:arcs]] <= ref[:branch][l]["rate_a"])
    for (l, i, j) in ref[:arcs]
        JuMP.set_name(p[(l, i, j)], "p_$(l)_$(i)_$(j)")
        JuMP.set_name(q[(l, i, j)], "q_$(l)_$(i)_$(j)")
    end
    JuMP.@objective(model, Min, sum(gen["cost"][1]*pg[i]^2 + gen["cost"][2]*pg[i] + gen["cost"][3] for (i,gen) in ref[:gen]))

    for (i,bus) in ref[:ref_buses]
        JuMP.@constraint(model, va[i] == 0)
    end

    if add_param_load
        @variable(model, pload[i in keys(ref[:bus])] ∈ MOI.Parameter.(0.0))
        @variable(model, qload[i in keys(ref[:bus])] ∈ MOI.Parameter.(0.0))
    else
        pload = zeros(length(ref[:bus]))
        qload = zeros(length(ref[:bus]))
    end

    demand_equilibrium = Dict{Int, JuMP.ConstraintRef}()
    for (i,bus) in ref[:bus]
        bus_loads = [ref[:load][l] for l in ref[:bus_loads][i]]
        bus_shunts = [ref[:shunt][s] for s in ref[:bus_shunts][i]]

        active_balance = JuMP.@constraint(model,
            sum(p[a] for a in ref[:bus_arcs][i]) ==
            sum(pg[g] for g in ref[:bus_gens][i]) -
            sum(load["pd"] for load in bus_loads) -
            pload[i] -
            sum(shunt["gs"] for shunt in bus_shunts)*vm[i]^2
        )

        JuMP.@constraint(model,
            sum(q[a] for a in ref[:bus_arcs][i]) ==
            sum(qg[g] for g in ref[:bus_gens][i]) -
            sum(load["qd"] for load in bus_loads) -
            qload[i] +
            sum(shunt["bs"] for shunt in bus_shunts)*vm[i]^2
        )

        demand_equilibrium[i] = active_balance
    end

    # Branch power flow physics and limit constraints
    for (i,branch) in ref[:branch]
        f_idx = (i, branch["f_bus"], branch["t_bus"])
        t_idx = (i, branch["t_bus"], branch["f_bus"])

        p_fr = p[f_idx]
        q_fr = q[f_idx]
        p_to = p[t_idx]
        q_to = q[t_idx]

        vm_fr = vm[branch["f_bus"]]
        vm_to = vm[branch["t_bus"]]
        va_fr = va[branch["f_bus"]]
        va_to = va[branch["t_bus"]]

        g, b = PowerModels.calc_branch_y(branch)
        tr, ti = PowerModels.calc_branch_t(branch)
        ttm = tr^2 + ti^2
        g_fr = branch["g_fr"]
        b_fr = branch["b_fr"]
        g_to = branch["g_to"]
        b_to = branch["b_to"]

        # From side of the branch flow
        JuMP.@constraint(model, p_fr ==  (g+g_fr)/ttm*vm_fr^2 + (-g*tr+b*ti)/ttm*(vm_fr*vm_to*cos(va_fr-va_to)) + (-b*tr-g*ti)/ttm*(vm_fr*vm_to*sin(va_fr-va_to)) )
        JuMP.@constraint(model, q_fr == -(b+b_fr)/ttm*vm_fr^2 - (-b*tr-g*ti)/ttm*(vm_fr*vm_to*cos(va_fr-va_to)) + (-g*tr+b*ti)/ttm*(vm_fr*vm_to*sin(va_fr-va_to)) )

        # To side of the branch flow
        JuMP.@constraint(model, p_to ==  (g+g_to)*vm_to^2 + (-g*tr-b*ti)/ttm*(vm_to*vm_fr*cos(va_to-va_fr)) + (-b*tr+g*ti)/ttm*(vm_to*vm_fr*sin(va_to-va_fr)) )
        JuMP.@constraint(model, q_to == -(b+b_to)*vm_to^2 - (-b*tr+g*ti)/ttm*(vm_to*vm_fr*cos(va_to-va_fr)) + (-g*tr-b*ti)/ttm*(vm_to*vm_fr*sin(va_to-va_fr)) )

        # Voltage angle difference limit
        JuMP.@constraint(model, branch["angmin"] <= va_fr - va_to <= branch["angmax"])

        # Apparent power limit, from side and to side
        JuMP.@constraint(model, p_fr^2 + q_fr^2 <= branch["rate_a"]^2)
        JuMP.@constraint(model, p_to^2 + q_to^2 <= branch["rate_a"]^2)
    end

    return model, ref, demand_equilibrium, p, q, pg, qg, va, vm, pload, qload
end

function build_bidding_opf_model(case_name; percen_bidding_nodes=0.1, solver=Ipopt.Optimizer)
    data = make_basic_network(pglib(case_name))
    data["basic_network"] = false
    # add bidding generators
    num_bidding_nodes = ceil(Int, length(data["bus"]) * percen_bidding_nodes)
    bidding_nodes = parse.(Int, rand(keys(data["bus"]), num_bidding_nodes))
    existing_gens = maximum(parse.(Int, collect(keys(data["gen"]))))
    pmax = maximum([data["gen"][g]["pmax"] for g in keys(data["gen"])])
    qmax = maximum([data["gen"][g]["qmax"] for g in keys(data["gen"])])
    bidding_gen_ids = existing_gens + 1:existing_gens + num_bidding_nodes
    for (i, node) in enumerate(bidding_nodes)
        data["gen"]["$(bidding_gen_ids[i])"] = build_gen(node, bidding_gen_ids[i], pmax, qmax)
    end
    data = make_basic_network(data)
    total_market = sum([data["gen"][g]["pmax"] for g in keys(data["gen"])])
    
    model, ref, demand_equilibrium, p, q, pg, qg, va, vm, pload, qload = build_opf_model(data; solver=solver)

    # add bids
    JuMP.@variable(model, pg_bid[i in bidding_gen_ids] ∈ MOI.Parameter.(pmax * 0.5))
    @constraint(model, bid[i in bidding_gen_ids], pg[i] <= pg_bid[i])

    model_variables = JuMP.num_variables(model)

    # for consistency with other solvers, skip the variable bounds in the constraint count
    model_constraints = JuMP.num_constraints(model; count_variable_in_set_constraints = false)

    println("")
    println("\033[1mSummary\033[0m")
    println("   case........: $(case_name)")
    println("   variables...: $(model_variables)")
    println("   constraints.: $(model_constraints)")

    println("")

    bidding_generators_dispatch = [pg[i] for i in bidding_gen_ids]
    _pg_bid = [pg_bid[i] for i in bidding_gen_ids]
    all_primal_variables = all_variables(model)
    remaining_vars = setdiff(all_primal_variables, bidding_generators_dispatch)
    remaining_vars = setdiff(remaining_vars, _pg_bid)
    return Dict(
        "case" => case_name,
        "model_variables" => remaining_vars,
        "bidding_generators_dispatch" => bidding_generators_dispatch,
        "bidding_lmps" => [demand_equilibrium[i] for i in bidding_nodes],
        "model" => model,
        "bid" => _pg_bid,
        "pmax" => pmax,
        "total_market" => total_market
    )
end

function build_bidding_upper(num_bidding_nodes, pmax; solver=Ipopt.Optimizer)
    upper_model = Model(solver)

    @variable(upper_model, 0 <= pg_bid[i=1:num_bidding_nodes] <= pmax)
    @variable(upper_model, pg[i=1:num_bidding_nodes])
    @variable(upper_model, lambda[i=1:num_bidding_nodes])
    @objective(upper_model, Max, dot(lambda, pg))
    return upper_model, lambda, pg_bid, pg
end

function memoize(foo::Function, n_outputs::Int)
    last_x, last_f = nothing, nothing
    function foo_i(i, x...)
        if x !== last_x
            ret = foo(x...)
            if !isnothing(ret)
                last_x, last_f = x, ret
            end
        end
        return last_f[i]
    end
    return [(x...) -> foo_i(i, x...) for i in 1:n_outputs]
end

function memoize!(foo::Function, n_outputs::Int)
    last_x, last_f = nothing, nothing
    function foo_i(i, g, x...)
        if x !== last_x
            ret = foo(g, x...)
            if !isnothing(ret)
                last_x, last_f = x, ret
            end
        end
        g .= last_f[i]
        return last_f[i]
    end
    return [(g, x...) -> foo_i(i, g, x...) for i in 1:n_outputs]
end

function fdiff_derivatives(f::Function)
    function ∇f(g::AbstractVector{T}, x::Vararg{T,N}) where {T,N}
        FiniteDiff.finite_difference_gradient!(g, y -> f(y...), collect(x))
        return
    end
    function ∇²f(H::AbstractMatrix{T}, x::Vararg{T,N}) where {T,N}
        h = FiniteDiff.finite_difference_hessian!(y -> f(y...), collect(x))
        for i in 1:N, j in 1:i
            H[i, j] = h[i, j]
        end
        return
    end
    return ∇f, ∇²f
end

function test_bilevel_ac_strategic_bidding(case_name="pglib_opf_case5_pjm.m"; percen_bidding_nodes=0.1, Δp=0.0001, solver_upper=Ipopt.Optimizer, solver_lower=Ipopt.Optimizer)
    # test derivative of the dual of the demand equilibrium constraint
    data = build_bidding_opf_model(case_name; percen_bidding_nodes=percen_bidding_nodes, solver=solver_lower)
    pmax = data["pmax"]
    primal_vars = [data["bidding_generators_dispatch"]; data["model_variables"]]
    num_bidding_nodes = length(data["bidding_generators_dispatch"])
    set_parameter_value.(data["bid"], 0.01)
    JuMP.optimize!(data["model"])
    evaluator, cons = create_evaluator(data["model"]; x=[primal_vars; data["bid"]])
    leq_locations, geq_locations = find_inequealities(cons)
    num_ineq = length(leq_locations) + length(geq_locations)
    num_primal = length(primal_vars)
    bidding_lmps_index = [findall(x -> x == i, cons)[1] for i in data["bidding_lmps"]]
    if !isnothing(Δp)
        Δp = fill(Δp, num_bidding_nodes)
    end
    # Δp = rand(-pmax*0.1:0.001:pmax*0.1, num_bidding_nodes)
    # Δs, sp = compute_sensitivity(evaluator, cons, Δp; primal_vars=primal_vars, params=data["bid"])
    
    # set_parameter_value.(data["bid"], 0.5 .+ Δp)
    # JuMP.optimize!(data["model"])

    # @test dual.(data["bidding_lmps"]) ≈ Δs[(num_primal + num_ineq) .+ bidding_lmps_index]

    # test bilevel strategic bidding
    upper_model, lambda, pg_bid, pg = build_bidding_upper(num_bidding_nodes, pmax; solver=solver_upper)
    evaluator, cons = create_evaluator(data["model"]; x=[primal_vars; data["bid"]])

    function f(pg_bid_val...)
        set_parameter_value.(data["bid"], pg_bid_val)
        JuMP.optimize!(data["model"])
        if !is_solved_and_feasible(data["model"])
            return nothing
        end
        return [value.(data["bidding_generators_dispatch"]); -dual.(data["bidding_lmps"])]
    end

    function ∇f(g::AbstractVector{T}, pg_bid_val...) where {T}
        if !all(value.(data["bid"]) .== pg_bid_val)
            set_parameter_value.(data["bid"], pg_bid_val)
            JuMP.optimize!(data["model"])
            @assert is_solved_and_feasible(data["model"])
        end
        
        Δs, sp = compute_sensitivity(evaluator, cons, Δp; primal_vars=primal_vars, params=data["bid"])
        if isnothing(Δp)
            Δs = Δs * ones(size(Δs, 2))
        end
        for i in 1:num_bidding_nodes
            g[i] = Δs[i]
        end
        for (i, b_idx) in enumerate(bidding_lmps_index)
            g[i] = -Δs[num_primal + num_ineq + b_idx]
        end
        return g
        # return [Δs[1:num_bidding_nodes]; -(Δs[(num_primal + num_ineq) .+ bidding_lmps_index])]
    end

    memoized_f = memoize(f, 2 * num_bidding_nodes)
    memoized_∇f = !isnothing(Δp) && iszero(Δp) ? [(args...) -> nothing for i in 1:2 * num_bidding_nodes] : memoize!(∇f, 2 * num_bidding_nodes)

    for i in 1:num_bidding_nodes
        op_pg = add_nonlinear_operator(upper_model, num_bidding_nodes, memoized_f[i], memoized_∇f[i]; name = Symbol("op_pg_$i"))
        # op_pg = add_nonlinear_operator(upper_model, num_bidding_nodes, memoized_f[i], fdiff_derivatives(memoized_f[i])...; name = Symbol("op_pg_$i"))
        @constraint(upper_model, pg[i] == op_pg(pg_bid...))
    end

    for i in 1:num_bidding_nodes
        op_lambda = add_nonlinear_operator(upper_model, num_bidding_nodes, memoized_f[num_bidding_nodes + i], memoized_∇f[num_bidding_nodes + i]; name = Symbol("op_lambda_$i"))
        # op_lambda = add_nonlinear_operator(upper_model, num_bidding_nodes, memoized_f[num_bidding_nodes + i], fdiff_derivatives(memoized_f[num_bidding_nodes + i])...; name = Symbol("op_lambda_$i"))
        @constraint(upper_model, lambda[i] == op_lambda(pg_bid...))
    end

    JuMP.optimize!(upper_model)

    println("Status: ", termination_status(upper_model))
    println("Objective: ", objective_value(upper_model))
    println("Duals: ", value.(lambda))
    println("Bids: ", value.(pg_bid))
    println("Dispatch: ", value.(pg))
end

function test_bidding_nlopt(case_name="pglib_opf_case5_pjm.m"; percen_bidding_nodes=0.1, Δp=0.0001, solver_upper=:LD_MMA, solver_lower=optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0), 
    max_eval=100, pmax_multiplier=1.0
)
    data = build_bidding_opf_model(case_name; percen_bidding_nodes=percen_bidding_nodes, solver=solver_lower)
    pmax = data["pmax"] * pmax_multiplier
    primal_vars = [data["bidding_generators_dispatch"]; data["model_variables"]]
    num_bidding_nodes = length(data["bidding_generators_dispatch"])
    set_parameter_value.(data["bid"], 0.01)
    JuMP.optimize!(data["model"])
    evaluator, cons = create_evaluator(data["model"]; x=[primal_vars; data["bid"]])
    leq_locations, geq_locations = find_inequealities(cons)
    num_ineq = length(leq_locations) + length(geq_locations)
    num_primal = length(primal_vars)
    bidding_lmps_index = [findall(x -> x == i, cons)[1] for i in data["bidding_lmps"]]
    if !isnothing(Δp)
        Δp = fill(Δp, num_bidding_nodes)
    end

    evaluator, cons = create_evaluator(data["model"]; x=[primal_vars; data["bid"]])

    function f(pg_bid_val...)
        set_parameter_value.(data["bid"], pg_bid_val)
        JuMP.optimize!(data["model"])
        @assert is_solved_and_feasible(data["model"])

        return value.(data["bidding_generators_dispatch"]), -dual.(data["bidding_lmps"])
    end

    function ∇f(pg_bid_val...)
        @assert all(value.(data["bid"]) .== pg_bid_val)
        
        Δs, sp = compute_sensitivity(evaluator, cons, Δp; primal_vars=primal_vars, params=data["bid"])
        if isnothing(Δp)
            Δs = Δs * ones(size(Δs, 2))
        end

        return Δs[1:num_bidding_nodes], -(Δs[(num_primal + num_ineq) .+ bidding_lmps_index])
    end

    if !isnothing(Δp) && iszero(Δp)
        _∇f = (pg_bid_val...) -> (zeros(num_bidding_nodes), zeros(length(bidding_lmps_index)))
    else
        _∇f = ∇f
    end

    trace = Any[]
    function my_objective_fn(pg_bid_val::Vector, grad::Vector)
        pg, lmps = f(pg_bid_val...)
        Δpg, Δlmps = _∇f(pg_bid_val...)
        value = dot(lmps, pg)
        if length(grad) > 0
            grad .= dot(Δlmps, pg) .+ dot(lmps, Δpg)
        end
        # println("Objective: ", value)
        push!(trace, copy(pg_bid_val) => value)
        return value
    end

    opt = NLopt.Opt(solver_upper, num_bidding_nodes)
    NLopt.lower_bounds!(opt, fill(0.0, num_bidding_nodes))
    NLopt.upper_bounds!(opt, fill(pmax, num_bidding_nodes))
    NLopt.xtol_rel!(opt, 1e-4)
    maxeval!(opt, max_eval)
    NLopt.max_objective!(opt, my_objective_fn)
    max_f, opt_x, ret = NLopt.optimize(opt, rand(num_bidding_nodes))
    num_evals = NLopt.numevals(opt)
    println("Status: ", ret)
    println("Objective: ", max_f)
    println("Bids: ", opt_x[1:num_bidding_nodes])
    println("Duals: ", opt_x[num_bidding_nodes+1:end])
    println("Number of evaluations: ", num_evals)
    return max_f, num_evals, trace, (sum(trace[end][1]) / data["total_market"]) * 100
end

function sesitivity_load(case_name="pglib_opf_case5_pjm.m"; Δp=nothing, solver=Ipopt.Optimizer)
    data = make_basic_network(pglib(case_name))

    model, ref, demand_equilibrium, p, q, pg, qg, va, vm, pload, qload = build_opf_model(data; solver=solver, add_param_load=true)
    num_bus = length(ref[:bus])
    demand_equilibrium = [demand_equilibrium[i] for i in 1:num_bus]

    JuMP.optimize!(model)

    params = vcat(pload.data, qload.data)
    all_primal_variables = all_variables(model)
    primal_vars = setdiff(all_primal_variables, params)

    evaluator, cons = create_evaluator(model; x=[primal_vars; params])

    leq_locations, geq_locations = find_inequealities(cons)
    num_ineq = length(leq_locations) + length(geq_locations)
    num_primal = length(primal_vars)
    lmps_index = [findall(x -> x == i, cons)[1] for i in demand_equilibrium]

    Δs, sp = compute_sensitivity(evaluator, cons, Δp; primal_vars=primal_vars, params=params)
    return Δs[1:num_primal, :], Δs[(num_primal + num_ineq) .+ lmps_index, :]
end
