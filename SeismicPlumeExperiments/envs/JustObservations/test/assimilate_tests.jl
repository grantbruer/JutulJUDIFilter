@testset "Test assimilate defined" begin
    ms = methods(EnsembleFilters.assimilate_data)
    filter_types = [tuple(m.sig.types[2:3]...) for m in ms]
    @test (EnsembleJustObsFilter, EnsembleJustObsFilter) in filter_types
end

@testset "Test scalar inversion" begin
    params = TOML.parsefile(joinpath(@__DIR__, "params_scalar.toml"));
    job_dir = mktempdir(".")
    work_dir = mktempdir(".")
    step_index = 1

    observer = x -> (x .+ 1)
    x_true = [2.0]
    y_obs = observer(x_true)
    x0 = zeros(size(x_true))

    x = EnsembleJustObsFilters.invert_observation(observer, y_obs, x0; params, job_dir, step_index, work_dir)
    @test x ≈ x_true
end

@testset "Test vector inversion" begin
    params = TOML.parsefile(joinpath(@__DIR__, "params_scalar.toml"));
    job_dir = mktempdir(".")
    work_dir = mktempdir(".")
    step_index = 1

    observer = x -> (x .+ 1)
    x_true = [1.0 2.0 3.0]
    y_obs = observer(x_true)
    x0 = zeros(size(x_true))

    x = EnsembleJustObsFilters.invert_observation(observer, y_obs, x0; params, job_dir, step_index, work_dir)
    @test x ≈ x_true
end

@testset "Test vertical gradient loss" begin
    params = TOML.parsefile(joinpath(@__DIR__, "params_scalar.toml"));
    job_dir = mktempdir(".")
    work_dir = mktempdir(".")
    step_index = 1

    observer = x -> (x .+ 1)
    x_true = [1.0 1.0 1.0; 2.0 2.0 2.0]
    x_true_noisy = [1.0 1.1 0.9; 2.0 2.1 1.9]
    @show x_true
    y_obs = observer(x_true)
    y_obs_noisy = observer(x_true_noisy)
    x0 = zeros(size(x_true))

    x = EnsembleJustObsFilters.invert_observation(observer, y_obs_noisy, x0; params, job_dir, step_index, work_dir)
    @test x ≈ x_true_noisy

    EnsembleJustObsFilters.get_vertical_spacing(::typeof(observer)) = 1e0
    params["filter"]["optimization"]["vertical_gradient_norm_type"] = "hybrid_l1_l2"
    params["filter"]["optimization"]["vertical_gradient_scale"] = 1e-3

    x = EnsembleJustObsFilters.invert_observation(observer, y_obs_noisy, x0; params, job_dir, step_index, work_dir)
    @test x ≈ x_true atol=0.06 norm=v->maximum(abs.(v))
end

@testset "Test horizontal gradient loss" begin
    params = TOML.parsefile(joinpath(@__DIR__, "params_scalar.toml"));
    job_dir = mktempdir(".")
    work_dir = mktempdir(".")
    step_index = 1

    observer = x -> (x .+ 1)
    x_true = [1.0 2.0 3.0; 1.0 2.0 3.0]
    x_true_noisy = [1.0 2.0 3.0; 0.9 2.1 3.1]
    @show x_true
    y_obs = observer(x_true)
    y_obs_noisy = observer(x_true_noisy)
    x0 = zeros(size(x_true))

    @test size(x0) == size(x_true_noisy)
    @test size(x0) == size(y_obs)

    x = EnsembleJustObsFilters.invert_observation(observer, y_obs_noisy, x0; params, job_dir, step_index, work_dir)
    @test x ≈ x_true_noisy

    EnsembleJustObsFilters.get_horizontal_spacing(::typeof(observer)) = 1e0
    params["filter"]["optimization"]["horizontal_gradient_norm_type"] = "hybrid_l1_l2"
    params["filter"]["optimization"]["horizontal_gradient_scale"] = 1e-2

    x = EnsembleJustObsFilters.invert_observation(observer, y_obs_noisy, x0; params, job_dir, step_index, work_dir)
    @test x ≈ x_true atol=0.06 norm=v->maximum(abs.(v))
end

# @testset "Test box constraint" begin
#     params = TOML.parsefile(joinpath(@__DIR__, "params_scalar.toml"));
#     job_dir = mktempdir(".")
#     work_dir = mktempdir(".")
#     step_index = 1

#     params["filter"]["optimization"]["constraints"] = "box"
#     params["filter"]["optimization"]["box_range"] = [-0.99, 0.99]

#     observer = x -> tan.(pi/2*x) .+ 1
#     x_true = [0.4]
#     y_obs = observer(x_true)
#     x0 = ones(size(x_true))

#     x = EnsembleJustObsFilters.invert_observation(observer, y_obs, x0; params, job_dir, step_index, work_dir)
#     @test x ≈ x_true
# end

# @testset "Test thresholding" begin
#     params = TOML.parsefile(joinpath(@__DIR__, "params_scalar.toml"));
#     job_dir = mktempdir(".")
#     work_dir = mktempdir(".")
#     step_index = 1

#     params["filter"]["optimization"]["constraints"] = "threshold"
#     params["filter"]["optimization"]["nonzero_threshold"] = 0.05

#     observer_points = linrange(0, 3, 20)
#     observer_points_pows = []
#     observer = x -> 
#     x_true = [0, 1, 0, -1/6, 0, 1/120]
#     # y_obs = sin.(observer_points)
#     y_obs = observer(x_true)
#     x0 = zeros(size(x_true))

#     x = EnsembleJustObsFilters.invert_observation(observer, y_obs, x0; params, job_dir, step_index, work_dir)
#     @test x ≈ x_true
# end
