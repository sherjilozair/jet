@[Link("curand")]
lib LibCURAND
  alias GeneratorSt = Void
  alias DistributionShiftSt = Void
  alias DistributionM2ShiftSt = Void
  alias HistogramM2St = Void
  alias DiscreteDistributionSt = Void
  fun create_generator = curandCreateGenerator(generator : GeneratorT*, rng_type : RngTypeT) : StatusT
  type GeneratorT = Void*
  enum RngType
    X_RngTest = 0
    X_RngPseudoDefault = 100
    X_RngPseudoXorwow = 101
    X_RngPseudoMrg32K3A = 121
    X_RngPseudoMtgp32 = 141
    X_RngPseudoMt19937 = 142
    X_RngPseudoPhilox43210 = 161
    X_RngQuasiDefault = 200
    X_RngQuasiSobol32 = 201
    X_RngQuasiScrambledSobol32 = 202
    X_RngQuasiSobol64 = 203
    X_RngQuasiScrambledSobol64 = 204
  end
  type RngTypeT = RngType
  enum Status
    X_StatusSuccess = 0
    X_StatusVersionMismatch = 100
    X_StatusNotInitialized = 101
    X_StatusAllocationFailed = 102
    X_StatusTypeError = 103
    X_StatusOutOfRange = 104
    X_StatusLengthNotMultiple = 105
    X_StatusDoublePrecisionRequired = 106
    X_StatusLaunchFailure = 201
    X_StatusPreexistingFailure = 202
    X_StatusInitializationFailed = 203
    X_StatusArchMismatch = 204
    X_StatusInternalError = 999
  end
  type StatusT = Status
  fun create_generator_host = curandCreateGeneratorHost(generator : GeneratorT*, rng_type : RngTypeT) : StatusT
  fun destroy_generator = curandDestroyGenerator(generator : GeneratorT) : StatusT
  fun get_version = curandGetVersion(version : LibC::Int*) : StatusT
  fun get_property = curandGetProperty(type : LibraryPropertyType, value : LibC::Int*) : StatusT
  enum LibraryPropertyTypeT
    MajorVersion = 0
    MinorVersion = 1
    PatchLevel = 2
  end
  type LibraryPropertyType = LibraryPropertyTypeT
  fun set_stream = curandSetStream(generator : GeneratorT, stream : CudaStreamT) : StatusT
  type CudaStreamT = Void*
  fun set_pseudo_random_generator_seed = curandSetPseudoRandomGeneratorSeed(generator : GeneratorT, seed : LibC::ULongLong) : StatusT
  fun set_generator_offset = curandSetGeneratorOffset(generator : GeneratorT, offset : LibC::ULongLong) : StatusT
  fun set_generator_ordering = curandSetGeneratorOrdering(generator : GeneratorT, order : OrderingT) : StatusT
  enum Ordering
    X_OrderingPseudoBest = 100
    X_OrderingPseudoDefault = 101
    X_OrderingPseudoSeeded = 102
    X_OrderingQuasiDefault = 201
  end
  type OrderingT = Ordering
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
  type DiscreteDistributionT = Void*
  fun destroy_distribution = curandDestroyDistribution(discrete_distribution : DiscreteDistributionT) : StatusT
  fun generate_poisson = curandGeneratePoisson(generator : GeneratorT, output_ptr : LibC::UInt*, n : LibC::Int, lambda : LibC::Double) : StatusT
  fun generate_poisson_method = curandGeneratePoissonMethod(generator : GeneratorT, output_ptr : LibC::UInt*, n : LibC::Int, lambda : LibC::Double, method : MethodT) : StatusT
  enum Method
    X_ChooseBest = 0
    X_Itr = 1
    X_Knuth = 2
    X_Hitr = 3
    X_M1 = 4
    X_M2 = 5
    X_BinarySearch = 6
    X_DiscreteGauss = 7
    X_Rejection = 8
    X_DeviceApi = 9
    X_FastRejection = 10
    X_3Rd = 11
    X_Definition = 12
    X_Poisson = 13
  end
  type MethodT = Method
  fun generate_binomial = curandGenerateBinomial(generator : GeneratorT, output_ptr : LibC::UInt*, num : LibC::Int, n : LibC::UInt, p : LibC::Double) : StatusT
  fun generate_binomial_method = curandGenerateBinomialMethod(generator : GeneratorT, output_ptr : LibC::UInt*, num : LibC::Int, n : LibC::UInt, p : LibC::Double, method : MethodT) : StatusT
  fun generate_seeds = curandGenerateSeeds(generator : GeneratorT) : StatusT
  fun get_direction_vectors32 = curandGetDirectionVectors32(vectors : DirectionVectors32T**, set : DirectionVectorSetT) : StatusT
  alias DirectionVectors32T = LibC::UInt[32]
  enum DirectionVectorSet
    X_DirectionVectors32Joekuo6 = 101
    X_ScrambledDirectionVectors32Joekuo6 = 102
    X_DirectionVectors64Joekuo6 = 103
    X_ScrambledDirectionVectors64Joekuo6 = 104
  end
  type DirectionVectorSetT = DirectionVectorSet
  fun get_scramble_constants32 = curandGetScrambleConstants32(constants : LibC::UInt**) : StatusT
  fun get_direction_vectors64 = curandGetDirectionVectors64(vectors : DirectionVectors64T**, set : DirectionVectorSetT) : StatusT
  alias DirectionVectors64T = LibC::ULongLong[64]
  fun get_scramble_constants64 = curandGetScrambleConstants64(constants : LibC::ULongLong**) : StatusT
end
