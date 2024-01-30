# NQCModels.jl

[![CI](https://github.com/NQCD/NQCModels.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/NQCD/NQCModels.jl/actions/workflows/CI.yml)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://nqcd.github.io/NQCDynamics.jl/dev/NQCModels/overview/)

This package provides an interface for defining models for nonadiabatic dynamics.
Primarily, the package is intended to be used with [NQCDynamics.jl](https://github.com/NQCD/NQCDynamics.jl/)
but it is equally possible to use any of the models separately.

For details on the included models and implementation take a look at the [documentation](https://nqcd.github.io/NQCDynamics.jl/dev/NQCModels/overview/).

## For Developer

If you wanted to test your local branch NQCModels.jl, you have to set the path of NQCModels.jl locally. Otherwise, you Julia kernel will automatically use you default NQCModels package. You can do it by 
```
(@v1.9) pkg> dev /Your-path/NQCModels.jl
```
Of course, if you wanted to free your local repo and switch back to default package, it can be done by
```
(@v1.9) pkg> free NQCModels
```

