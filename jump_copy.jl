using JuMP.MOI
function add_par_model_vars!(model::JuMP.Model, par_model::JuMP.Model, var_src_to_dest::Dict{VariableRef, VariableRef})
    allvars = all_variables(par_model)
    x = @variable(model, [1:length(allvars)])
    for (src, dest) in zip(allvars, x)
        var_src_to_dest[src] = dest
        JuMP.set_name(dest, JuMP.name(src))
    end
    return var_src_to_dest
end


function copy_and_replace_variables(
    src::Vector,
    map::Dict{JuMP.VariableRef,JuMP.VariableRef},
)
    return copy_and_replace_variables.(src, Ref(map))
end

function copy_and_replace_variables(
    src::Real,
    ::Dict{JuMP.VariableRef,JuMP.VariableRef},
)
    return src
end

function copy_and_replace_variables(
    src::JuMP.VariableRef,
    src_to_dest_variable::Dict{JuMP.VariableRef,JuMP.VariableRef},
)
    return src_to_dest_variable[src]
end

function copy_and_replace_variables(
    src::JuMP.GenericAffExpr,
    src_to_dest_variable::Dict{JuMP.VariableRef,JuMP.VariableRef},
)
    return JuMP.GenericAffExpr(
        src.constant,
        Pair{VariableRef,Float64}[
            src_to_dest_variable[key] => val for (key, val) in src.terms
        ],
    )
end

function copy_and_replace_variables(
    src::JuMP.GenericQuadExpr,
    src_to_dest_variable::Dict{JuMP.VariableRef,JuMP.VariableRef},
)
    return JuMP.GenericQuadExpr(
        copy_and_replace_variables(src.aff, src_to_dest_variable),
        Pair{UnorderedPair{VariableRef},Float64}[
            UnorderedPair{VariableRef}(
                src_to_dest_variable[pair.a],
                src_to_dest_variable[pair.b],
            ) => coef for (pair, coef) in src.terms
        ],
    )
end

function copy_and_replace_variables(
    src::JuMP.GenericNonlinearExpr{V},
    src_to_dest_variable::Dict{JuMP.VariableRef,JuMP.VariableRef},
) where {V}
    num_args = length(src.args)
    args = Vector{Any}(undef, num_args)
    for i = 1:num_args
        args[i] = copy_and_replace_variables(src.args[i], src_to_dest_variable)
    end

    return @expression(owner_model(first(src_to_dest_variable)[2]), eval(src.head)(args...))
end

function copy_and_replace_variables(
    src::Any,
    ::Dict{JuMP.VariableRef,JuMP.VariableRef},
)
    return error(
        "`copy_and_replace_variables` is not implemented for functions like `$(src)`.",
    )
end

function create_constraint(model, obj, var_src_to_dest)
    new_func = copy_and_replace_variables(obj.func, var_src_to_dest)
    return @constraint(model, new_func in obj.set)
end

function create_constraint(model, obj::ScalarConstraint{NonlinearExpr, MOI.EqualTo{Float64}}, var_src_to_dest)
    new_func = copy_and_replace_variables(obj.func, var_src_to_dest)
    return @constraint(model, new_func == obj.set.value)
end

function create_constraint(model, obj::ScalarConstraint{NonlinearExpr, MOI.LessThan{Float64}}, var_src_to_dest)
    new_func = copy_and_replace_variables(obj.func, var_src_to_dest)
    return @constraint(model, new_func <= obj.set.upper)
end

function create_constraint(model, obj::ScalarConstraint{NonlinearExpr, MOI.GreaterThan{Float64}}, var_src_to_dest)
    new_func = copy_and_replace_variables(obj.func, var_src_to_dest)
    return @constraint(model, new_func >= obj.set.lower)
end

function add_child_model_exps!(model::JuMP.Model, par_model::JuMP.Model, var_src_to_dest::Dict{VariableRef, VariableRef})
    # Add constraints:
    cons_to_cons = Dict()
    for (F, S) in JuMP.list_of_constraint_types(par_model)
        S <: MathOptInterface.Parameter && continue
        for con in JuMP.all_constraints(par_model, F, S)
            obj = JuMP.constraint_object(con)
            cons_to_cons[con] = create_constraint(model, obj, var_src_to_dest)
        end
    end
    # Add objective:
    current = JuMP.objective_function(model)
    par_model_objective =
        copy_and_replace_variables(JuMP.objective_function(par_model), var_src_to_dest)
    JuMP.set_objective_function(
        model,
        current + par_model_objective,
    )
    return cons_to_cons
end


function copy_jump_no_parameters(par_model::JuMP.Model, model = JuMP.Model())
    set_objective_sense(model, objective_sense(par_model))
    var_src_to_dest = Dict{VariableRef, VariableRef}()
    add_par_model_vars!(model, par_model, var_src_to_dest)

    cons_to_cons = add_child_model_exps!(model, par_model, var_src_to_dest)

    return model, var_src_to_dest, cons_to_cons
end