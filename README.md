
# JutulJUDIFilter

Data assimilation using fluid flow by [Jutul.jl], [JutulDarcy.jl], and [JutulDarcyRules.jl] with seismic observations from [JUDI.jl].

## Layout

This repo is organized into 1 runnable project (SeismicPlumeExperiments) and 6 modules.

- EnsembleFilters: defines an abstract interface for working with ensembles.
- EnsembleJustObsFilters: implements the interface for an ensemble that uses a least-squares inversion.
- EnsembleKalmanFilters: implements the interface for an ensemble that uses Kalman filtering.
- EnsembleNormalizingFlowFilters: implements the interface for an ensemble that uses normalizing flow.
- MyUtils: implements functions I like to use.
- SeismicPlumeEnsembleFilter: defines the transition and observation operators for use on ensemble members.

SeismicPlumeExperiments has the environments and scripts for doing data assimilation.

## Developer documentation

TODO: For now, just message me if you want to add a filter, extend some functionality, add a parameter, etc.


[Jutul.jl]:https://github.com/sintefmath/Jutul.jl
[JutulDarcy.jl]:https://github.com/sintefmath/JutulDarcy.jl
[JutulDarcyRules.jl]:https://github.com/slimgroup/JutulDarcyRules.jl
[JUDI.jl]:https://github.com/slimgroup/JUDI.jl

