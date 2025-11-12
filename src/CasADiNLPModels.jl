module CasADiNLPModels
using NLPModels
using Libdl, SparseArrays, LinearAlgebra, JSON

export CasADiFunction, CasADiNLPModel

include("lib_managment.jl")
include("casadi_function.jl")
include("casadi_nlp.jl")

end
