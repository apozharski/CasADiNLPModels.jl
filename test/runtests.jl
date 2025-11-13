using CasADiNLPModels
using LinearAlgebra
using NLPModels
using Test

# TODO(@anton) add more tests as necessary
include("lib_management/refcount.jl")

@testset "CasADiNLPModels API" begin
    abs_so = abspath(joinpath(@__DIR__, "nlp.so"))
    abs_json = abspath(joinpath(@__DIR__, "nlp.json"))
    nlp = CasADiNLPModel(abs_so, abs_json)
    n, m = NLPModels.get_nvar(nlp), NLPModels.get_ncon(nlp)

    x0 = NLPModels.get_x0(nlp)
    y0 = NLPModels.get_y0(nlp)
    @test isa(x0, Vector{Float64})
    @test isa(y0, Vector{Float64})
    @test length(x0) == n
    @test length(y0) == m
    # Objective
    @test NLPModels.obj(nlp, x0) isa Float64
    # Constraints
    c = zeros(Float64, m)
    NLPModels.cons!(nlp, x0, c)
    # Gradient
    g = zeros(Float64, n)
    NLPModels.grad!(nlp, x0, g)
    # Jacobian
    nnzj = NLPModels.get_nnzj(nlp)
    Ji, Jj = NLPModels.jac_structure(nlp)
    @test isa(Ji, Vector{Int})
    @test isa(Jj, Vector{Int})
    @test length(Ji) == length(Jj) == nnzj
    Jx = zeros(Float64, nnzj)
    NLPModels.jac_coord!(nlp, x0, Jx)
    # Hessian
    nnzh = NLPModels.get_nnzh(nlp)
    Hi, Hj = NLPModels.hess_structure(nlp)
    @test isa(Hi, Vector{Int})
    @test isa(Hj, Vector{Int})
    @test length(Hi) == length(Hj) == nnzh
    Hx = zeros(Float64, nnzh)
    NLPModels.hess_coord!(nlp, x0, y0, Hx)
end

