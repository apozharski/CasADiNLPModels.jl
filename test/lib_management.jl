@testset "Test Deconflicting" begin
    abs_so = abspath(joinpath(@__DIR__, "nlp.$(dlext)"))
    abs_json = abspath(joinpath(@__DIR__, "nlp.json"))
    nlp1 = CasADiNLPModel(abs_so, abs_json)
    nlp2 = CasADiNLPModel(abs_so, abs_json)
    @test nlp1.lib != nlp2.lib
    @test dlpath(nlp1.lib) != dlpath(nlp2.lib)
    @test nlp1.f._eval != nlp2.f._eval
end
