@testset "Test Refcounting logic" begin
    @testset "basic refcount" begin
        # Absolute paths
        abs_so = abspath(joinpath(@__DIR__, "..", "nlp.$(dlext)"))
        abs_json = abspath(joinpath(@__DIR__, "..", "nlp.json"))
        # Relative paths
        rel_so = joinpath(@__DIR__, "..", "nlp.$(dlext)")
        rel_json = joinpath(@__DIR__, "..", "nlp.json")
        nlp = CasADiNLPModel(abs_so, abs_json)
        nlp2 = CasADiNLPModel(rel_so, rel_json)
        lib = nlp.lib
        @test CasADiNLPModels.lib_refcount[lib][1] == abs_so
        @test CasADiNLPModels.lib_refcount[lib][2] == 12

        # Pretend to GC
        finalize(nlp)
        finalize(nlp.f)
        finalize(nlp.grad_f)
        finalize(nlp.g)
        finalize(nlp.jac_g)
        finalize(nlp.hess_L)
        @test CasADiNLPModels.lib_refcount[lib][1] == abs_so
        @test CasADiNLPModels.lib_refcount[lib][2] == 6

        # Pretend to GC
        finalize(nlp2)
        finalize(nlp2.f)
        finalize(nlp2.grad_f)
        finalize(nlp2.g)
        finalize(nlp2.jac_g)
        finalize(nlp2.hess_L)
        @test !haskey(CasADiNLPModels.lib_refcount, lib)
    end
end
