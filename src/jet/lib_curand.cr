@[Link("curand")]
lib LibCURAND
  alias GeneratorSt = Void
  alias DistributionShiftSt = Void
  alias DistributionM2ShiftSt = Void
  alias HistogramM2St = Void
  alias DiscreteDistributionSt = Void
  alias LibraryPropertyType = LibraryPropertyTypeT
  alias RngTypeT = RngType
  alias StatusT = Status
  alias OrderingT = Ordering
  alias DirectionVectors64T = LibC::ULongLong[64]
  alias DirectionVectors32T = LibC::UInt[32]
  alias DirectionVectorSetT = DirectionVectorSet
  alias MethodT = Method

  type GeneratorT = Void*
  type DiscreteDistributionT = Void*
  type CudaStreamT = Void*

  enum RngType
    RngTest                  =   0
    RngPseudoDefault         = 100
    RngPseudoXorwow          = 101
    RngPseudoMrg32K3A        = 121
    RngPseudoMtgp32          = 141
    RngPseudoMt19937         = 142
    RngPseudoPhilox43210     = 161
    RngQuasiDefault          = 200
    RngQuasiSobol32          = 201
    RngQuasiScrambledSobol32 = 202
    RngQuasiSobol64          = 203
    RngQuasiScrambledSobol64 = 204
  end

  enum Status
    Success                 =   0
    VersionMismatch         = 100
    NotInitialized          = 101
    AllocationFailed        = 102
    TypeError               = 103
    OutOfRange              = 104
    LengthNotMultiple       = 105
    DoublePrecisionRequired = 106
    LaunchFailure           = 201
    PreexistingFailure      = 202
    InitializationFailed    = 203
    ArchMismatch            = 204
    InternalError           = 999
  end

  enum LibraryPropertyTypeT
    MajorVersion = 0
    MinorVersion = 1
    PatchLevel   = 2
  end

  enum Ordering
    OrderingPseudoBest    = 100
    OrderingPseudoDefault = 101
    OrderingPseudoSeeded  = 102
    OrderingQuasiDefault  = 201
  end

  enum Method
    ChooseBest    =  0
    Itr           =  1
    Knuth         =  2
    Hitr          =  3
    M1            =  4
    M2            =  5
    BinarySearch  =  6
    DiscreteGauss =  7
    Rejection     =  8
    DeviceApi     =  9
    FastRejection = 10
    X_3Rd         = 11
    Definition    = 12
    Poisson       = 13
  end

  enum DirectionVectorSet
    DirectionVectors32Joekuo6          = 101
    ScrambledDirectionVectors32Joekuo6 = 102
    DirectionVectors64Joekuo6          = 103
    ScrambledDirectionVectors64Joekuo6 = 104
  end

  fun set_stream = curandSetStream(generator : GeneratorT, stream : CudaStreamT) : StatusT
  fun create_generator = curandCreateGenerator(generator : GeneratorT*, rng_type : RngTypeT) : StatusT
  fun create_generator_host = curandCreateGeneratorHost(generator : GeneratorT*, rng_type : RngTypeT) : StatusT
  fun destroy_generator = curandDestroyGenerator(generator : GeneratorT) : StatusT
  fun get_version = curandGetVersion(version : LibC::Int*) : StatusT
  fun get_property = curandGetProperty(type : LibraryPropertyType, value : LibC::Int*) : StatusT
  fun set_pseudo_random_generator_seed = curandSetPseudoRandomGeneratorSeed(generator : GeneratorT, seed : LibC::ULongLong) : StatusT
  fun set_generator_offset = curandSetGeneratorOffset(generator : GeneratorT, offset : LibC::ULongLong) : StatusT
  fun set_generator_ordering = curandSetGeneratorOrdering(generator : GeneratorT, order : OrderingT) : StatusT
  fun set_quasi_random_generator_dimensions = curandSetQuasiRandomGeneratorDimensions(generator : GeneratorT, num_dimensions : LibC::UInt) : StatusT
  fun generate = curandGenerate(generator : GeneratorT, output_ptr : LibC::UInt*, num : LibC::Int) : StatusT
  fun generate_long_long = curandGenerateLongLong(generator : GeneratorT, output_ptr : LibC::ULongLong*, num : LibC::Int) : StatusT
  fun generate_uniform = curandGenerateUniform(generator : GeneratorT, output_ptr : LibC::Float*, num : LibC::Int) : StatusT
  fun generate_uniform_double = curandGenerateUniformDouble(generator : GeneratorT, output_ptr : LibC::Double*, num : LibC::Int) : StatusT
  fun generate_normal = curandGenerateNormal(generator : GeneratorT, output_ptr : LibC::Float*, n : LibC::Int, mean : LibC::Float, stddev : LibC::Float) : StatusT
  fun generate_normal_double = curandGenerateNormalDouble(generator : GeneratorT, output_ptr : LibC::Double*, n : LibC::Int, mean : LibC::Double, stddev : LibC::Double) : StatusT
  fun generate_log_normal = curandGenerateLogNormal(generator : GeneratorT, output_ptr : LibC::Float*, n : LibC::Int, mean : LibC::Float, stddev : LibC::Float) : StatusT
  fun generate_log_normal_double = curandGenerateLogNormalDouble(generator : GeneratorT, output_ptr : LibC::Double*, n : LibC::Int, mean : LibC::Double, stddev : LibC::Double) : StatusT
  fun create_poisson_distribution = curandCreatePoissonDistribution(lambda : LibC::Double, discrete_distribution : DiscreteDistributionT*) : StatusT
  fun destroy_distribution = curandDestroyDistribution(discrete_distribution : DiscreteDistributionT) : StatusT
  fun generate_poisson = curandGeneratePoisson(generator : GeneratorT, output_ptr : LibC::UInt*, n : LibC::Int, lambda : LibC::Double) : StatusT
  fun generate_poisson_method = curandGeneratePoissonMethod(generator : GeneratorT, output_ptr : LibC::UInt*, n : LibC::Int, lambda : LibC::Double, method : MethodT) : StatusT
  fun generate_binomial = curandGenerateBinomial(generator : GeneratorT, output_ptr : LibC::UInt*, num : LibC::Int, n : LibC::UInt, p : LibC::Double) : StatusT
  fun generate_binomial_method = curandGenerateBinomialMethod(generator : GeneratorT, output_ptr : LibC::UInt*, num : LibC::Int, n : LibC::UInt, p : LibC::Double, method : MethodT) : StatusT
  fun generate_seeds = curandGenerateSeeds(generator : GeneratorT) : StatusT
  fun get_direction_vectors32 = curandGetDirectionVectors32(vectors : DirectionVectors32T**, set : DirectionVectorSetT) : StatusT
  fun get_scramble_constants32 = curandGetScrambleConstants32(constants : LibC::UInt**) : StatusT
  fun get_direction_vectors64 = curandGetDirectionVectors64(vectors : DirectionVectors64T**, set : DirectionVectorSetT) : StatusT
  fun get_scramble_constants64 = curandGetScrambleConstants64(constants : LibC::ULongLong**) : StatusT
end
