using JuMP.MOI
using SparseArrays

function create_nlp_model(model::JuMP.Model)
    rows = Any[]
    nlp = MOI.Nonlinear.Model()
    for (F, S) in list_of_constraint_types(model)
        if F <: VariableRef
            continue  # Skip variable bounds
        end
        for ci in all_constraints(model, F, S)
            push!(rows, ci)
            object = constraint_object(ci)
            MOI.Nonlinear.add_constraint(nlp, object.func, object.set)
        end
    end
    MOI.Nonlinear.set_objective(nlp, objective_function(model))
    return nlp, rows
end

function fill_off_diagonal(H)
    ret = H + H'
    row_vals = SparseArrays.rowvals(ret)
    non_zeros = SparseArrays.nonzeros(ret)
    for col in 1:size(ret, 2)
        for i in SparseArrays.nzrange(ret, col)
            if col == row_vals[i]
                non_zeros[i] /= 2
            end
        end
    end
    return ret
end

function compute_optimal_hessian(model::Model; nlp=nothing, rows=nothing, x=nothing, evaluator=nothing, backend=nothing
)
    if isnothing(nlp)
        nlp, rows = create_nlp_model(model)
        x=all_variables(model), 
        backend=MOI.Nonlinear.SparseReverseMode()
        evaluator = MOI.Nonlinear.Evaluator(nlp, backend, index.(x))
        MOI.initialize(evaluator, [:Hess])
    end

    hessian_sparsity = MOI.hessian_lagrangian_structure(evaluator)
    I = [i for (i, _) in hessian_sparsity]
    J = [j for (_, j) in hessian_sparsity]
    V = zeros(length(hessian_sparsity))
    MOI.eval_hessian_lagrangian(evaluator, V, value.(x), 1.0, dual.(rows))
    H = SparseArrays.sparse(I, J, V, length(x), length(x))
    return Matrix(fill_off_diagonal(H))
end

function compute_optimal_jacobian(model::Model; nlp=nothing, rows=nothing, x=nothing, evaluator=nothing, backend=nothing
)
    if isnothing(nlp)
        nlp, rows = create_nlp_model(model)
        x=all_variables(model), 
        backend=MOI.Nonlinear.SparseReverseMode()
        evaluator = MOI.Nonlinear.Evaluator(nlp, backend, index.(x))
        MOI.initialize(evaluator, [:Jac])
    end
    jacobian_sparsity = MOI.jacobian_structure(evaluator)
    I = [i for (i, _) in jacobian_sparsity]
    J = [j for (_, j) in jacobian_sparsity]
    V = zeros(length(jacobian_sparsity))
    MOI.eval_constraint_jacobian(evaluator, V, value.(x))
    A = SparseArrays.sparse(I, J, V, length(rows), length(x))
    return Matrix(A)
end

function compute_optimal_hess_jac(model::Model; nlp_rows=create_nlp_model(model), x=all_variables(model), 
    backend=MOI.Nonlinear.SparseReverseMode(), 
    evaluator = MOI.Nonlinear.Evaluator(nlp_rows[1], backend, index.(x))
)
    MOI.initialize(evaluator, [:Hess, :Jac])
    hessian = compute_optimal_hessian(model, nlp=nlp_rows[1], rows=nlp_rows[2], x=x, backend=backend, evaluator=evaluator)
    jacobian = compute_optimal_jacobian(model, nlp=nlp_rows[1], rows=nlp_rows[2], x=x, backend=backend, evaluator=evaluator)
    
    return hessian, jacobian, nlp_rows[1], nlp_rows[2]
end



