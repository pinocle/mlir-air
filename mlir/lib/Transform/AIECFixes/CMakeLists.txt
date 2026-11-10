# Included by aie-compiler, whose generated aiec pass declarations are shared
# with its frontend and device passes. The upstream air-opt is unchanged.
add_library(AIECAIRFixes STATIC
  PruneTokenLoops_fix.cpp
  SynchronizeHost_fix.cpp
  Residency_fix.cpp
  ProjectLaunch_fix.cpp
  GroupPrograms_fix.cpp
  EliminateStagingBuffers_fix.cpp
  FuseConvBlocks_fix.cpp
)
add_dependencies(AIECAIRFixes AIECPassIncGen AIRHeaders)
