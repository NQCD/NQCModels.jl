using Test
using NQCBase
using NQCModels

include("test_utils.jl")

atoms = Atoms([:H, :Pt, :Pt, :Pt, :Pt])
cell = PeriodicCell([10 0 0; 0 10 0; 0 0 10])
lib_path = "path/to/md_tian2_lib.so"
pes_path = "path/to/EMT-HPt.pes"

model = NQCModels.AdiabaticModels.md_tian2_EMT(atoms, cell, lib_path, pes_path)
@test test_model(model, 5)

