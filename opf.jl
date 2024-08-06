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

function build_opf(case_name; percen_bidding_nodes=0.1)
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
    # create ref
    PowerModels.standardize_cost_terms!(data, order=2)
    PowerModels.calc_thermal_limits!(data)
    ref = PowerModels.build_ref(data)[:it][:pm][:nw][0]

    # Model
    model = JuMP.Model(Ipopt.Optimizer)
    #JuMP.set_optimizer_attribute(model, "print_level", 0)

    JuMP.@variable(model, va[i in keys(ref[:bus])])
    JuMP.@variable(model, ref[:bus][i]["vmin"] <= vm[i in keys(ref[:bus])] <= ref[:bus][i]["vmax"], start=1.0)
    for i in keys(ref[:bus])
        JuMP.set_name(va[i], "va_$i")
        JuMP.set_name(vm[i], "vm_$i")
    end
    JuMP.@variable(model, ref[:gen][i]["pmin"] <= pg[i in keys(ref[:gen])] <= ref[:gen][i]["pmax"])
    JuMP.@variable(model, ref[:gen][i]["qmin"] <= qg[i in keys(ref[:gen])] <= ref[:gen][i]["qmax"])

    # add bids
    JuMP.@variable(model, pg_bid[i in bidding_gen_ids] ∈ MOI.Parameter.(1.0))
    @constraint(model, bid[i in bidding_gen_ids], pg[i] <= pg_bid[i])

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

    demand_equilibrium = Dict{Int, JuMP.ConstraintRef}()
    for (i,bus) in ref[:bus]
        bus_loads = [ref[:load][l] for l in ref[:bus_loads][i]]
        bus_shunts = [ref[:shunt][s] for s in ref[:bus_shunts][i]]

        active_balance = JuMP.@constraint(model,
            sum(p[a] for a in ref[:bus_arcs][i]) ==
            sum(pg[g] for g in ref[:bus_gens][i]) -
            sum(load["pd"] for load in bus_loads) -
            sum(shunt["gs"] for shunt in bus_shunts)*vm[i]^2
        )

        JuMP.@constraint(model,
            sum(q[a] for a in ref[:bus_arcs][i]) ==
            sum(qg[g] for g in ref[:bus_gens][i]) -
            sum(load["qd"] for load in bus_loads) +
            sum(shunt["bs"] for shunt in bus_shunts)*vm[i]^2
        )

        demand_equilibrium[bus] = active_balance
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

    model_variables = JuMP.num_variables(model)

    # for consistency with other solvers, skip the variable bounds in the constraint count
    model_constraints = JuMP.num_constraints(model; count_variable_in_set_constraints = false)

    println("")
    println("\033[1mSummary\033[0m")
    println("   case........: $(file_name)")
    println("   variables...: $(model_variables)")
    println("   constraints.: $(model_constraints)")

    println("")

    return Dict(
        "case" => file_name,
        "model_variables" => model_variables,
        "bidding_generators_dispatch" => [pg[i] for i in bidding_gen_ids],
        "bidding_lmps" => [demand_equilibrium[i] for i in bidding_nodes],
        "model" => model,
        "bid" => [pg_bid[i] for i in bidding_gen_ids],
        "pmax" => pmax,
    )
end

function build_bidding_upper(num_bidding_nodes, pmax)
    upper_model = Model(Ipopt.Optimizer)

    @variable(upper_model, 0 <= pg_bid[i=1:num_bidding_nodes] <= pmax)
    @variable(upper_model, lambda[i=1:num_bidding_nodes])
    @objective(upper_model, Max, dot(lambda, pg_bid))
    return upper_model, lambda, pg_bid
end
