# 
using Pkg
Pkg.activate(@__DIR__)
try

	using Distributed, ClusterManagers
catch
	Pkg.add("ClusterManagers")
	Pkg.checkout("ClusterManagers")
end

using Distributed, ClusterManagers

np = 5 #
addprocs(SlurmManager(np), job_file_loc = ARGS[1])

println("We are all connected and ready.")

include(ARGS[2])