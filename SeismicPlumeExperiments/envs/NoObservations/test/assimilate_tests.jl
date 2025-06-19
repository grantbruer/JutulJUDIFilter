@testset "Test assimilate defined" begin
    ms = methods(EnsembleFilters.assimilate_data)
    filter_types = [tuple(m.sig.types[2:3]...) for m in ms]
    @test (EnsembleNoObsFilter, EnsembleNoObsFilter) in filter_types
end
