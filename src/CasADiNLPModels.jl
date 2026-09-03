module CasADiNLPModels
using NLPModels
using Libdl, SparseArrays, LinearAlgebra, JSON, Random

export CasADiFunction, CasADiNLPModel
export eval!

include("lib_management.jl")
include("casadi_function.jl")
include("casadi_nlp.jl")

end
