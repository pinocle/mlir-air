// AIE1 project extensions to MLIR-AIR, consumed by aiec-opt.
#ifndef AIEC_AIR_FIX_PASSES
#define AIEC_AIR_FIX_PASSES

include "mlir/Pass/PassBase.td"

def AIECPruneTokenLoops : Pass<"aiec-prune-token-loops", "ModuleOp"> {
  let summary = "Remove segment loops that only carry async tokens";
  let description = [{
    air-isolate-async-dma-loop-nests{scope=launch} moves a segment's DMA loop
    nests to the launch and leaves their token skeletons in the segment:
    scf.for / scf.parallel loops whose bodies hold only air.wait_all, nested
    loops and constants. Such a loop has no effect, and native air-to-aie
    (29e3586) fails to legalize the air.wait_all of an scf.parallel around a
    doubly nested scf.for of them. Each such outermost loop outside a herd is
    replaced by an air.wait_all on its initial tokens, which keeps every
    ordering it expressed.
  }];
  let dependentDialects = ["xilinx::air::airDialect", "mlir::scf::SCFDialect"];
}

def AIECEliminateStagingBuffers : Pass<"aiec-eliminate-staging-buffers", "ModuleOp"> {
  let summary = "Let herds move data from L3 directly where a buffer only stages it";
  let description = [{
    The Triton-XDNA transform scripts keep a program's whole block in an L2
    buffer: triton-shared loads into an allocation, flatten_tile_forall
    allocates the result in memory space 1, and each core's DMAs move its part
    between that buffer and L1. AIE1 (xcvc1902) has no L2 memory, and
    air-to-aie cannot place those buffers.

    After the transform script (bufferized, herds formed, before
    air-wrap-func-with-parallel), an allocation outside herds that only
    stages data is removed:
      - written by one memref.copy from a non-allocation source and read only
        by herds whose tied argument is a DMA source: the herds take the
        source;
      - written only by herds through DMA destinations and read by one
        memref.copy to a non-allocation destination after them: the herds take
        the destination, whose view ops are moved before the first herd.
    Operations between the copy and the herds must not write memory other than
    allocations (and, for outputs, not read the destination). Ported from
    hexagon-mlir IntermediateBufferEliminationPattern (CopyCanonicalizationPass).

    The staging buffer was a snapshot of the whole block. Without it, a core
    reads its input when it runs, so two pointer arguments to overlapping
    memory are only correct when each core reads what it alone writes.
  }];
  let dependentDialects = ["xilinx::air::airDialect", "mlir::memref::MemRefDialect"];
}

def AIECSynchronizeHost : Pass<"aiec-synchronize-host", "ModuleOp"> {
  let summary = "Run the host side of an async AIR schedule in program order";
  let description = [{
    Program order is a topological order of the tokens aircc's air-dependency
    puts on the host side: air.execute bodies are inlined, every host token
    becomes a complete token (an air.wait_all with no dependencies at the
    start of its isolated scope: the function or a launch), and scf.parallel
    loops that only combine tokens lose
    their reductions. Device segments keep their tokens; air-to-aie compiles
    them. With strip-hierarchy-tokens, launches, segments and host channel
    endpoints also lose their (complete) dependencies and token results, the
    sequential host ABI the runtime executes.
  }];
  let dependentDialects = ["xilinx::air::airDialect", "mlir::scf::SCFDialect"];
  let options = [
    Option<"stripHierarchyTokens", "strip-hierarchy-tokens", "bool", "false",
           "Also remove dependencies and token results of launches, segments and host channel endpoints">
  ];
}

def AIECFuseConvBlocks : Pass<"aiec-fuse-conv-blocks", "ModuleOp"> {
  let summary = "Fuse resident convolutions with their accumulator's initialization and epilogue";
  let description = [{
    On the memref IR the resident conv Transform leaves (one scf.forall per
    convolution, the accumulator tile promoted to L1), as cuDNN/CUTLASS conv
    epilogues, XLA fusions and the Vitis AI DPU do:

    * Prologue: an accumulator that starts as a copy of a broadcast of a 1-D
      tensor along the output-channel dimension starts, in each core, as that
      tensor's tile broadcast in L1; the full-size broadcast goes away.
    * Epilogue: the accumulator's only other reader, an all-parallel
      identity-map elementwise op over buffers like the accumulator, runs on
      the tile in L1 (its other inputs' tiles loaded beside it) and writes its
      own output; or an expand_shape + transposing copy, written by the core
      directly into the permuted destination when the split tile divides the
      inner dimension and starts on a tile boundary.

    A part is applied only when the core still fits its tile: L1 buffers with
    stack and static data within the module, input kinds within the tile's
    and its north neighbour's S2MM channels, and the herd's streams within the
    transport's budget. The report lists the fused operations' markers and,
    per kept part, the reason.
  }];
  let dependentDialects = ["mlir::arith::ArithDialect", "mlir::linalg::LinalgDialect",
                           "mlir::memref::MemRefDialect", "mlir::scf::SCFDialect"];
  let options = [
    Option<"moduleBytes", "module-bytes", "int64_t", "32768", "Tile memory module bytes">,
    Option<"stackBytes", "stack-bytes", "int64_t", "2048", "Core stack in the module">,
    Option<"staticBytes", "static-bytes", "int64_t", "512", "Core static data in the module">,
    Option<"channelsPerDirection", "channels-per-direction", "int64_t", "2", "Tile DMA S2MM channels">,
    Option<"streamTotal", "stream-total", "int64_t", "0", "Streams per program (packet ids); 0: unbounded">,
    Option<"streamInputs", "stream-inputs", "int64_t", "0", "Input streams per program; 0: unbounded">,
    Option<"streamOutputs", "stream-outputs", "int64_t", "0", "Output streams per program; 0: unbounded">,
    Option<"report", "report", "std::string", "\"\"", "JSON report path">
  ];
}

def AIECAnnotateUniformInit : Pass<"aiec-annotate-uniform-init", "ModuleOp"> {
  let summary = "Mark Linalg outputs whose whole buffer holds a proven constant";
  let description = [{
    A conservative reaching-definition analysis over each function's
    top-level operations: a linalg.fill of a whole buffer with an
    arith.constant, possibly copied whole into another non-parameter buffer,
    reaches a Linalg operation whose output region argument is read. That
    operation gets aie1.uniform_output_init (one FloatAttr or unit per
    output) for aiec-localize-accumulators. Any other Linalg write, unknown
    operation, call or control flow kills every fact, and a copy into a
    function parameter kills all of them (parameters may alias). Existing
    annotations from the input are removed first.
  }];
  let dependentDialects = ["mlir::linalg::LinalgDialect"];
}

def AIECLocalizeAccumulators : Pass<"aiec-localize-accumulators", "ModuleOp"> {
  let summary = "Initialize promoted reduction outputs in local memory from proven uniform values";
  let description = [{
    The contract between native Linalg promotion (air-linalg-codegen with
    l1-promote) and AIR channel lowering. An operation carrying
    aie1.uniform_output_init (a proven whole-buffer constant per output, see
    aiec_air.cnn_fix) has its promoted L1 output loaded by one memref.copy and
    stored by another. The load becomes a linalg.fill of that constant, so
    no input DMA brings the initial value. Before that, the copy pair is
    hoisted out of enclosing static positive scf.for loops that do not
    produce the output view and have no other access to it (AIR's
    HoistReduceBufferPattern, for synchronous copies). A loop that still
    encloses the load must vary the output tile, or the fill would reset a
    partial sum. Anything else is an error. The scalar computation is not
    changed. The report lists each localized output.
  }];
  let dependentDialects = ["mlir::arith::ArithDialect", "mlir::linalg::LinalgDialect",
                           "mlir::memref::MemRefDialect", "mlir::scf::SCFDialect"];
  let options = [Option<"report", "report", "std::string", "\"\"", "JSON report path">];
}

def AIECSplitReadWriteFifo : Pass<"aiec-split-readwrite-fifo", "ModuleOp"> {
  let summary = "Give a herd buffer that is both received and sent distinct FIFO owners";
  let description = [{
    AIR's ObjectFIFO conversion replaces an allocation by its acquire result.
    A local buffer used by both a channel get and a channel put would receive
    two incompatible replacements. The get then lands in a new local buffer
    and a linalg.copy right after it fills the original. The get must be
    synchronous, full-buffer and in a herd or static scf.for body. If any
    buffer is split, named Linalg operations are generalized
    (linalg-generalize-named-ops), because AIR scalarizes named L1 copies.
  }];
  let dependentDialects = ["mlir::linalg::LinalgDialect", "mlir::memref::MemRefDialect"];
  let options = [Option<"report", "report", "std::string", "\"\"", "JSON report path">];
}

def AIECProjectLaunch : Pass<"aiec-project-launch", "ModuleOp"> {
  let summary = "Keep one air.launch of a mixed module as its compile projection";
  let description = [{
    Programs compile separately so that air-to-aie's module-local packet
    allocator starts fresh for each. The projection of launch `index` (in
    program order) erases every other launch, replacing its completion token
    by an already-complete air.wait_all, and the air.channel declarations
    the kept launch does not use. The host function, its ABI, side effects
    and constant initialization stay; canonicalize, cse and symbol-dce
    afterwards remove what became unused. The original module, not any
    projection, is the program.
  }];
  let options = [Option<"index", "index", "int64_t", "-1", "Launch to keep, in program order">];
}

def AIECGroupPrograms : Pass<"aiec-group-programs", "ModuleOp"> {
  let summary = "Form spatial AIR programs and replace proven DDR edges with device channels";
  let description = [{
    Runs on the sequential host form (aiec-synchronize-host with
    strip-hierarchy-tokens). Consecutive synchronous unit launches of a
    function become one launch and segment holding all their herds when the
    group fits the array (first-fit herd placement), the GMIO endpoints, the
    native packet allocator and the tiles' memory and locks. An intermediate
    allocation between two launches of a group, written by one host channel
    get and read by one put, becomes a device edge when its frames are
    proven: the same signature renames the consumer's channel to the
    producer's; the same ordered stream per slot with other frame boundaries,
    or a spatial retiling by element address identity (core-local pack and
    unpack over new packet channels), uses the packet transport. A full
    producer overwrite makes an earlier private host initialization dead.
    Host effects, escapes, aliasing, fanout and unproved layouts stay program
    boundaries with their reasons in the report. Device herds keep their
    async graph. This changes the mixed IR, not a side execution plan.
  }];
  let dependentDialects = ["mlir::affine::AffineDialect", "mlir::arith::ArithDialect",
                           "mlir::memref::MemRefDialect", "mlir::scf::SCFDialect"];
  let options = [
    Option<"columns", "columns", "int64_t", "50", "Array columns for herds">,
    Option<"rows", "rows", "int64_t", "8", "Core rows for herds">,
    Option<"inputPorts", "input-ports", "int64_t", "32", "GMIO input endpoints">,
    Option<"outputPorts", "output-ports", "int64_t", "32", "GMIO output endpoints">,
    Option<"localBytes", "local-bytes", "int64_t", "32768", "Tile memory module bytes">,
    Option<"stackBytes", "stack-bytes", "int64_t", "2048", "Core stack in the module">,
    Option<"locksPerTile", "locks-per-tile", "int64_t", "16", "Locks of a tile">,
    Option<"packetIds", "packet-ids", "int64_t", "32", "Native AIR packet IDs per device program">,
    Option<"retileReservedBytes", "retile-reserved-bytes", "int64_t", "1024",
           "Tile bytes kept for request control and allocator alignment when retiling">,
    Option<"report", "report", "std::string", "\"\"", "JSON report path">
  ];
}

#endif // AIEC_AIR_FIX_PASSES
