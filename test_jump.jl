model = JuMP.Model(Ipopt.Optimizer)

@variable(model, x)
@variable(model, y)

@objective(model, Min, x + y)

con1_ = @constraint(model, x >= 1)

con2_ = @constraint(model, -y <= -2)

JuMP.optimize!(model)

model.moi_backend.optimizer.model.inner.mult_g
model.moi_backend.optimizer.model.inner.g

dual(con1_)

dual(con2_)

#######

model = JuMP.Model(Ipopt.Optimizer)

@variable(model, x)
@variable(model, y)
@variable(model, s1)
@variable(model, s2)


@objective(model, Min, x + y)

con1 = @constraint(model, x - 1 - s1 == 0)

con2 = @constraint(model, -y + 2 - s2 == 0)

con1s = @constraint(model, s1 >= 0)
con2s = @constraint(model, s2 <= 0)

JuMP.optimize!(model)

dual(con1)
dual(con2)
dual(con1s)
dual(con2s)

#######

model = JuMP.Model(Ipopt.Optimizer)

@variable(model, x)
@variable(model, y)

@objective(model, Max, x + y)

con1_ = @constraint(model, -x >= -1)

con2_ = @constraint(model, y <= 2)

JuMP.optimize!(model)

model.moi_backend.optimizer.model.inner.mult_g
model.moi_backend.optimizer.model.inner.g

dual(con1_)

dual(con2_)

#######

model = JuMP.Model(Ipopt.Optimizer)

@variable(model, 0 <= x <= 1)
@variable(model, 0 <= y <= 1)

@objective(model, Max, x - y)

JuMP.optimize!(model)

model.moi_backend.optimizer.model.inner.mult_x_L
model.moi_backend.optimizer.model.inner.mult_x_U

dual.(LowerBoundRef(y))

dual.(UpperBoundRef(x))


#######

model = JuMP.Model(Ipopt.Optimizer)

@variable(model, 0 <= x <= 1)
@variable(model, 0 <= y <= 1)

@objective(model, Min, x - y)

JuMP.optimize!(model)

model.moi_backend.optimizer.model.inner.mult_x_L

model.moi_backend.optimizer.model.inner.mult_x_U

dual.(LowerBoundRef(x))

dual.(UpperBoundRef(y))

#######

model = JuMP.Model(Ipopt.Optimizer)

@variable(model, x)
@variable(model, y <= 2)
@variable(model, z >= 3)
@variable(model, w)
@variable(model, u)

@objective(model, Min, x - y + z + w - u)

con1 = @constraint(model, x - 1 == 0)

con1s = @constraint(model, w - 4 >= 0)
con2s = @constraint(model, u - 5 <= 0)

JuMP.optimize!(model)

JuMP.termination_status(model)

model.moi_backend.optimizer.model.inner.mult_x_L
model.moi_backend.optimizer.model.inner.mult_x_U

model.moi_backend.optimizer.model.inner.mult_g
model.moi_backend.optimizer.model.inner.g

dual.(UpperBoundRef(y))
dual.(LowerBoundRef(z))
dual(con1)
dual(con1s)
dual(con2s)


####### Max 

model = JuMP.Model(Ipopt.Optimizer)

@variable(model, x)
@variable(model, y <= 2)
@variable(model, z >= 3)
@variable(model, w)
@variable(model, u)

@objective(model, Max, - x + y - z - w + u)

con1 = @constraint(model, x - 1 == 0)
# s = w - 4 -> s >= 0 λ
con1s = @constraint(model, w - 4 >= 0)
con2s = @constraint(model, u - 5 <= 0)

JuMP.optimize!(model)

JuMP.termination_status(model)

model.moi_backend.optimizer.model.inner.mult_x_L
model.moi_backend.optimizer.model.inner.mult_x_U

model.moi_backend.optimizer.model.inner.mult_g
model.moi_backend.optimizer.model.inner.g

dual.(UpperBoundRef(y))
dual.(LowerBoundRef(z))
dual(con1)
dual(con1s)
dual(con2s)