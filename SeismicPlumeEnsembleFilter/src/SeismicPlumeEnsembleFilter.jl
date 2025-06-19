module SeismicPlumeEnsembleFilter

include("patchy.jl")
export find_water_bottom_immutable
export compute_patchy_constants
export Patchy

include("seismic_model.jl")
export SI_to_JUDI, JUDI_to_SI
export SeismicModel, SeismicModel_params
export build_source_receiver_geometry
export compute_noise_info, generate_noise

include("observation_model.jl")
export PatchyModel
export SeismicCO2Observer

include("plume_model.jl")
export PlumeModel, PlumeModelSetup
export get_permeability, set_permeability!

include("seismic_plume_ensemble.jl")
export load_uncertain_params
export transition_filter, observe_filter

include("seismic_plume_params.jl")
export get_observer
export get_porosity
export get_permeability_porosity
export get_velocity
export get_density
export get_velocity_density
export get_background_velocity_density
export initialize_plume_model
export SeismicModel_params

end # module
