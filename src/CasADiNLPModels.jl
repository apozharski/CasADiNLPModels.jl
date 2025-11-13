module CasADiNLPModels
using NLPModels
using Libdl, SparseArrays, LinearAlgebra, JSON

export CasADiFunction, CasADiNLPModel

include("lib_management.jl")
include("casadi_function.jl")
include("casadi_nlp.jl")

end
