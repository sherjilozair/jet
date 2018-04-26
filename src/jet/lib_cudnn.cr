@[Link("cudnn")]
lib LibCuDNN
  # Contants
  MAJOR          =    7
  MINOR          =    1
  PATCHLEVEL     =    2
  DIM_MAX        =    8
  LRN_MIN_N      =    1
  LRN_MAX_N      =   16
  LRN_MIN_K      = 1e-5
  LRN_MIN_BETA   = 0.01
  BN_MIN_EPSILON = 1e-5

  # Alias
  alias CallbackT = (SeverityT, Void*, DebugT*, LibC::Char* -> Void)
  alias Context = Void
  alias TensorStruct = Void
  alias ConvolutionStruct = Void
  alias PoolingStruct = Void
  alias FilterStruct = Void
  alias LrnStruct = Void
  alias ActivationStruct = Void
  alias SpatialTransformerStruct = Void
  alias OpTensorStruct = Void
  alias ReduceTensorStruct = Void
  alias CtcLossStruct = Void
  alias DropoutStruct = Void
  alias AlgorithmStruct = Void
  alias AlgorithmPerformanceStruct = Void
  alias RnnStruct = Void
  alias PersistentRnnPlan = Void

  # Types
  type HandleT = Void*
  type RuntimeTagT = Void*
  type CudaStreamT = Void*
  type TensorDescriptorT = Void*
  type OpTensorDescriptorT = Void*
  type ReduceTensorDescriptorT = Void*
  type ConvolutionDescriptorT = Void*
  type FilterDescriptorT = Void*
  type ActivationDescriptorT = Void*
  type PoolingDescriptorT = Void*
  type LrnDescriptorT = Void*
  type DropoutDescriptorT = Void*
  type SpatialTransformerDescriptorT = Void*
  type AlgorithmPerformanceT = Void*
  type RnnDescriptorT = Void*
  type PersistentRnnPlanT = Void*
  type AlgorithmDescriptorT = Void*
  type CtcLossDescriptorT = Void*

  # Enums
  enum StatusT
    Success                    =  0
    NotInitialized             =  1
    AllocFailed                =  2
    BadParam                   =  3
    InternalError              =  4
    InvalidValue               =  5
    ArchMismatch               =  6
    MappingError               =  7
    ExecutionFailed            =  8
    NotSupported               =  9
    LicenseError               = 10
    RuntimePrerequisiteMissing = 11
    RuntimeInProgress          = 12
    RuntimeFpOverflow          = 13
  end

  enum ErrQueryModeT
    ErrqueryRawcode     = 0
    ErrqueryNonblocking = 1
    ErrqueryBlocking    = 2
  end

  enum LibraryPropertyTypeT
    MajorVersion = 0
    MinorVersion = 1
    PatchLevel   = 2
  end

  enum TensorFormatT
    Nchw      = 0
    Nhwc      = 1
    NchwVectC = 2
  end

  enum DataTypeT
    Float   = 0
    Double  = 1
    Half    = 2
    Int8    = 3
    Int32   = 4
    Int8x4  = 5
    Uint8   = 6
    Uint8x4 = 7
  end

  enum OpTensorOpT
    Add  = 0
    Mul  = 1
    Min  = 2
    Max  = 3
    Sqrt = 4
    Not  = 5
  end

  enum NanPropagationT
    NotPropagateNan = 0
    PropagateNan    = 1
  end

  enum ReduceTensorOpT
    Add        = 0
    Mul        = 1
    Min        = 2
    Max        = 3
    Amax       = 4
    Avg        = 5
    Norm1      = 6
    Norm2      = 7
    MulNoZeros = 8
  end

  enum ReduceTensorIndicesT
    NoIndices        = 0
    FlattenedIndices = 1
  end

  enum IndicesTypeT
    A32BitIndices = 0
    A64BitIndices = 1
    A16BitIndices = 2
    A8BitIndices  = 3
  end

  enum ConvolutionModeT
    Convolution      = 0
    CrossCorrelation = 1
  end

  enum MathTypeT
    DefaultMath  = 0
    TensorOpMath = 1
  end

  enum ConvolutionFwdAlgoT
    ConvolutionFwdAlgoImplicitGemm        = 0
    ConvolutionFwdAlgoImplicitPrecompGemm = 1
    ConvolutionFwdAlgoGemm                = 2
    ConvolutionFwdAlgoDirect              = 3
    ConvolutionFwdAlgoFft                 = 4
    ConvolutionFwdAlgoFftTiling           = 5
    ConvolutionFwdAlgoWinograd            = 6
    ConvolutionFwdAlgoWinogradNonfused    = 7
    ConvolutionFwdAlgoCount               = 8
  end

  enum DeterminismT
    NonDeterministic = 0
    Deterministic    = 1
  end

  enum ConvolutionBwdDataAlgoT
    ConvolutionBwdDataAlgo0                = 0
    ConvolutionBwdDataAlgo1                = 1
    ConvolutionBwdDataAlgoFft              = 2
    ConvolutionBwdDataAlgoFftTiling        = 3
    ConvolutionBwdDataAlgoWinograd         = 4
    ConvolutionBwdDataAlgoWinogradNonfused = 5
    ConvolutionBwdDataAlgoCount            = 6
  end

  enum ConvolutionFwdPreferenceT
    ConvolutionFwdNoWorkspace           = 0
    ConvolutionFwdPreferFastest         = 1
    ConvolutionFwdSpecifyWorkspaceLimit = 2
  end

  enum ConvolutionBwdFilterPreferenceT
    ConvolutionBwdFilterNoWorkspace           = 0
    ConvolutionBwdFilterPreferFastest         = 1
    ConvolutionBwdFilterSpecifyWorkspaceLimit = 2
  end
  enum ConvolutionBwdFilterAlgoT
    ConvolutionBwdFilterAlgo0                = 0
    ConvolutionBwdFilterAlgo1                = 1
    ConvolutionBwdFilterAlgoFft              = 2
    ConvolutionBwdFilterAlgo3                = 3
    ConvolutionBwdFilterAlgoWinograd         = 4
    ConvolutionBwdFilterAlgoWinogradNonfused = 5
    ConvolutionBwdFilterAlgoFftTiling        = 6
    ConvolutionBwdFilterAlgoCount            = 7
  end

  enum ConvolutionBwdDataPreferenceT
    ConvolutionBwdDataNoWorkspace           = 0
    ConvolutionBwdDataPreferFastest         = 1
    ConvolutionBwdDataSpecifyWorkspaceLimit = 2
  end

  enum SoftmaxAlgorithmT
    SoftmaxFast     = 0
    SoftmaxAccurate = 1
    SoftmaxLog      = 2
  end

  enum SoftmaxModeT
    SoftmaxModeInstance = 0
    SoftmaxModeChannel  = 1
  end

  enum PoolingModeT
    PoolingMax                        = 0
    PoolingAverageCountIncludePadding = 1
    PoolingAverageCountExcludePadding = 2
    PoolingMaxDeterministic           = 3
  end

  enum ActivationModeT
    ActivationSigmoid     = 0
    ActivationRelu        = 1
    ActivationTanh        = 2
    ActivationClippedRelu = 3
    ActivationElu         = 4
    ActivationIdentity    = 5
  end

  enum DivNormModeT
    DivnormPrecomputedMeans = 0
  end

  enum LrnModeT
    LrnCrossChannelDim1 = 0
  end

  enum BatchNormModeT
    BatchnormPerActivation     = 0
    BatchnormSpatial           = 1
    BatchnormSpatialPersistent = 2
  end

  enum SamplerTypeT
    SamplerBilinear = 0
  end

  enum RnnInputModeT
    LinearInput = 0
    SkipInput   = 1
  end

  enum DirectionModeT
    Unidirectional = 0
    Bidirectional  = 1
  end

  enum RnnModeT
    RnnRelu = 0
    RnnTanh = 1
    Lstm    = 2
    Gru     = 3
  end

  enum RnnAlgoT
    RnnAlgoStandard       = 0
    RnnAlgoPersistStatic  = 1
    RnnAlgoPersistDynamic = 2
    RnnAlgoCount          = 3
  end

  enum SeverityT
    X_SevFatal   = 0
    X_SevError   = 1
    X_SevWarning = 2
    X_SevInfo    = 3
  end

  enum CtcLossAlgoT
    X_CtcLossAlgoDeterministic    = 0
    X_CtcLossAlgoNonDeterministic = 1
  end

  # Structs
  struct ConvolutionFwdAlgoPerfT
    algo : ConvolutionFwdAlgoT
    status : StatusT
    time : LibC::Float
    memory : LibC::Int
    determinism : DeterminismT
    math_type : MathTypeT
    reserved : LibC::Int[3]
  end

  struct ConvolutionBwdFilterAlgoPerfT
    algo : ConvolutionBwdFilterAlgoT
    status : StatusT
    time : LibC::Float
    memory : LibC::Int
    determinism : DeterminismT
    math_type : MathTypeT
    reserved : LibC::Int[3]
  end

  struct ConvolutionBwdDataAlgoPerfT
    algo : ConvolutionBwdDataAlgoT
    status : StatusT
    time : LibC::Float
    memory : LibC::Int
    determinism : DeterminismT
    math_type : MathTypeT
    reserved : LibC::Int[3]
  end

  struct DebugT
    _version : LibC::UInt
    status : StatusT
    time_sec : LibC::UInt
    time_usec : LibC::UInt
    time_delta : LibC::UInt
    handle : HandleT
    stream : CudaStreamT
    pid : LibC::ULongLong
    tid : LibC::ULongLong
    reserved : LibC::Int[16]
  end

  struct AlgorithmT
    algo : Algorithm
  end

  union Algorithm
    conv_fwd_algo : ConvolutionFwdAlgoT
    conv_bwd_filter_algo : ConvolutionBwdFilterAlgoT
    conv_bwd_data_algo : ConvolutionBwdDataAlgoT
    rnn_algo : RnnAlgoT
    ctc_loss_algo : CtcLossAlgoT
  end

  # functions
  fun get_version = cudnnGetVersion : LibC::Int
  fun get_cudart_version = cudnnGetCudartVersion : LibC::Int
  fun get_error_string = cudnnGetErrorString(status : StatusT) : LibC::Char*
  fun query_runtime_error = cudnnQueryRuntimeError(handle : HandleT, rstatus : StatusT*, mode : ErrQueryModeT, tag : RuntimeTagT) : StatusT
  fun get_property = cudnnGetProperty(type : LibraryPropertyTypeT, value : LibC::Int*) : StatusT
  fun create = cudnnCreate(handle : HandleT*) : StatusT
  fun destroy = cudnnDestroy(handle : HandleT) : StatusT
  fun set_stream = cudnnSetStream(handle : HandleT, stream_id : CudaStreamT) : StatusT
  fun get_stream = cudnnGetStream(handle : HandleT, stream_id : CudaStreamT*) : StatusT
  fun create_tensor_descriptor = cudnnCreateTensorDescriptor(tensor_desc : TensorDescriptorT*) : StatusT
  fun set_tensor4d_descriptor = cudnnSetTensor4dDescriptor(tensor_desc : TensorDescriptorT, format : TensorFormatT, data_type : DataTypeT, n : LibC::Int, c : LibC::Int, h : LibC::Int, w : LibC::Int) : StatusT
  fun set_tensor4d_descriptor_ex = cudnnSetTensor4dDescriptorEx(tensor_desc : TensorDescriptorT, data_type : DataTypeT, n : LibC::Int, c : LibC::Int, h : LibC::Int, w : LibC::Int, n_stride : LibC::Int, c_stride : LibC::Int, h_stride : LibC::Int, w_stride : LibC::Int) : StatusT
  fun get_tensor4d_descriptor = cudnnGetTensor4dDescriptor(tensor_desc : TensorDescriptorT, data_type : DataTypeT*, n : LibC::Int*, c : LibC::Int*, h : LibC::Int*, w : LibC::Int*, n_stride : LibC::Int*, c_stride : LibC::Int*, h_stride : LibC::Int*, w_stride : LibC::Int*) : StatusT
  fun set_tensor_nd_descriptor = cudnnSetTensorNdDescriptor(tensor_desc : TensorDescriptorT, data_type : DataTypeT, nb_dims : LibC::Int, dim_a : LibC::Int*, stride_a : LibC::Int*) : StatusT
  fun set_tensor_nd_descriptor_ex = cudnnSetTensorNdDescriptorEx(tensor_desc : TensorDescriptorT, format : TensorFormatT, data_type : DataTypeT, nb_dims : LibC::Int, dim_a : LibC::Int*) : StatusT
  fun get_tensor_nd_descriptor = cudnnGetTensorNdDescriptor(tensor_desc : TensorDescriptorT, nb_dims_requested : LibC::Int, data_type : DataTypeT*, nb_dims : LibC::Int*, dim_a : LibC::Int*, stride_a : LibC::Int*) : StatusT
  fun get_tensor_size_in_bytes = cudnnGetTensorSizeInBytes(tensor_desc : TensorDescriptorT, size : LibC::Int*) : StatusT
  fun destroy_tensor_descriptor = cudnnDestroyTensorDescriptor(tensor_desc : TensorDescriptorT) : StatusT
  fun transform_tensor = cudnnTransformTensor(handle : HandleT, alpha : Void*, x_desc : TensorDescriptorT, x : Void*, beta : Void*, y_desc : TensorDescriptorT, y : Void*) : StatusT
  fun add_tensor = cudnnAddTensor(handle : HandleT, alpha : Void*, a_desc : TensorDescriptorT, a : Void*, beta : Void*, c_desc : TensorDescriptorT, c : Void*) : StatusT
  fun create_op_tensor_descriptor = cudnnCreateOpTensorDescriptor(op_tensor_desc : OpTensorDescriptorT*) : StatusT
  fun set_op_tensor_descriptor = cudnnSetOpTensorDescriptor(op_tensor_desc : OpTensorDescriptorT, op_tensor_op : OpTensorOpT, op_tensor_comp_type : DataTypeT, op_tensor_nan_opt : NanPropagationT) : StatusT
  fun get_op_tensor_descriptor = cudnnGetOpTensorDescriptor(op_tensor_desc : OpTensorDescriptorT, op_tensor_op : OpTensorOpT*, op_tensor_comp_type : DataTypeT*, op_tensor_nan_opt : NanPropagationT*) : StatusT
  fun destroy_op_tensor_descriptor = cudnnDestroyOpTensorDescriptor(op_tensor_desc : OpTensorDescriptorT) : StatusT
  fun op_tensor = cudnnOpTensor(handle : HandleT, op_tensor_desc : OpTensorDescriptorT, alpha1 : Void*, a_desc : TensorDescriptorT, a : Void*, alpha2 : Void*, b_desc : TensorDescriptorT, b : Void*, beta : Void*, c_desc : TensorDescriptorT, c : Void*) : StatusT
  fun create_reduce_tensor_descriptor = cudnnCreateReduceTensorDescriptor(reduce_tensor_desc : ReduceTensorDescriptorT*) : StatusT
  fun set_reduce_tensor_descriptor = cudnnSetReduceTensorDescriptor(reduce_tensor_desc : ReduceTensorDescriptorT, reduce_tensor_op : ReduceTensorOpT, reduce_tensor_comp_type : DataTypeT, reduce_tensor_nan_opt : NanPropagationT, reduce_tensor_indices : ReduceTensorIndicesT, reduce_tensor_indices_type : IndicesTypeT) : StatusT
  fun get_reduce_tensor_descriptor = cudnnGetReduceTensorDescriptor(reduce_tensor_desc : ReduceTensorDescriptorT, reduce_tensor_op : ReduceTensorOpT*, reduce_tensor_comp_type : DataTypeT*, reduce_tensor_nan_opt : NanPropagationT*, reduce_tensor_indices : ReduceTensorIndicesT*, reduce_tensor_indices_type : IndicesTypeT*) : StatusT
  fun destroy_reduce_tensor_descriptor = cudnnDestroyReduceTensorDescriptor(reduce_tensor_desc : ReduceTensorDescriptorT) : StatusT
  fun get_reduction_indices_size = cudnnGetReductionIndicesSize(handle : HandleT, reduce_tensor_desc : ReduceTensorDescriptorT, a_desc : TensorDescriptorT, c_desc : TensorDescriptorT, size_in_bytes : LibC::Int*) : StatusT
  fun get_reduction_workspace_size = cudnnGetReductionWorkspaceSize(handle : HandleT, reduce_tensor_desc : ReduceTensorDescriptorT, a_desc : TensorDescriptorT, c_desc : TensorDescriptorT, size_in_bytes : LibC::Int*) : StatusT
  fun reduce_tensor = cudnnReduceTensor(handle : HandleT, reduce_tensor_desc : ReduceTensorDescriptorT, indices : Void*, indices_size_in_bytes : LibC::Int, workspace : Void*, workspace_size_in_bytes : LibC::Int, alpha : Void*, a_desc : TensorDescriptorT, a : Void*, beta : Void*, c_desc : TensorDescriptorT, c : Void*) : StatusT
  fun set_tensor = cudnnSetTensor(handle : HandleT, y_desc : TensorDescriptorT, y : Void*, value_ptr : Void*) : StatusT
  fun scale_tensor = cudnnScaleTensor(handle : HandleT, y_desc : TensorDescriptorT, y : Void*, alpha : Void*) : StatusT
  fun create_filter_descriptor = cudnnCreateFilterDescriptor(filter_desc : FilterDescriptorT*) : StatusT
  fun set_filter4d_descriptor = cudnnSetFilter4dDescriptor(filter_desc : FilterDescriptorT, data_type : DataTypeT, format : TensorFormatT, k : LibC::Int, c : LibC::Int, h : LibC::Int, w : LibC::Int) : StatusT
  fun get_filter4d_descriptor = cudnnGetFilter4dDescriptor(filter_desc : FilterDescriptorT, data_type : DataTypeT*, format : TensorFormatT*, k : LibC::Int*, c : LibC::Int*, h : LibC::Int*, w : LibC::Int*) : StatusT
  fun set_filter_nd_descriptor = cudnnSetFilterNdDescriptor(filter_desc : FilterDescriptorT, data_type : DataTypeT, format : TensorFormatT, nb_dims : LibC::Int, filter_dim_a : LibC::Int*) : StatusT
  fun get_filter_nd_descriptor = cudnnGetFilterNdDescriptor(filter_desc : FilterDescriptorT, nb_dims_requested : LibC::Int, data_type : DataTypeT*, format : TensorFormatT*, nb_dims : LibC::Int*, filter_dim_a : LibC::Int*) : StatusT
  fun destroy_filter_descriptor = cudnnDestroyFilterDescriptor(filter_desc : FilterDescriptorT) : StatusT
  fun create_convolution_descriptor = cudnnCreateConvolutionDescriptor(conv_desc : ConvolutionDescriptorT*) : StatusT
  fun set_convolution_math_type = cudnnSetConvolutionMathType(conv_desc : ConvolutionDescriptorT, math_type : MathTypeT) : StatusT
  fun get_convolution_math_type = cudnnGetConvolutionMathType(conv_desc : ConvolutionDescriptorT, math_type : MathTypeT*) : StatusT
  fun set_convolution_group_count = cudnnSetConvolutionGroupCount(conv_desc : ConvolutionDescriptorT, group_count : LibC::Int) : StatusT
  fun get_convolution_group_count = cudnnGetConvolutionGroupCount(conv_desc : ConvolutionDescriptorT, group_count : LibC::Int*) : StatusT
  fun set_convolution2d_descriptor = cudnnSetConvolution2dDescriptor(conv_desc : ConvolutionDescriptorT, pad_h : LibC::Int, pad_w : LibC::Int, u : LibC::Int, v : LibC::Int, dilation_h : LibC::Int, dilation_w : LibC::Int, mode : ConvolutionModeT, compute_type : DataTypeT) : StatusT
  fun get_convolution2d_descriptor = cudnnGetConvolution2dDescriptor(conv_desc : ConvolutionDescriptorT, pad_h : LibC::Int*, pad_w : LibC::Int*, u : LibC::Int*, v : LibC::Int*, dilation_h : LibC::Int*, dilation_w : LibC::Int*, mode : ConvolutionModeT*, compute_type : DataTypeT*) : StatusT
  fun get_convolution2d_forward_output_dim = cudnnGetConvolution2dForwardOutputDim(conv_desc : ConvolutionDescriptorT, input_tensor_desc : TensorDescriptorT, filter_desc : FilterDescriptorT, n : LibC::Int*, c : LibC::Int*, h : LibC::Int*, w : LibC::Int*) : StatusT
  fun set_convolution_nd_descriptor = cudnnSetConvolutionNdDescriptor(conv_desc : ConvolutionDescriptorT, array_length : LibC::Int, pad_a : LibC::Int*, filter_stride_a : LibC::Int*, dilation_a : LibC::Int*, mode : ConvolutionModeT, compute_type : DataTypeT) : StatusT
  fun get_convolution_nd_descriptor = cudnnGetConvolutionNdDescriptor(conv_desc : ConvolutionDescriptorT, array_length_requested : LibC::Int, array_length : LibC::Int*, pad_a : LibC::Int*, stride_a : LibC::Int*, dilation_a : LibC::Int*, mode : ConvolutionModeT*, compute_type : DataTypeT*) : StatusT
  fun get_convolution_nd_forward_output_dim = cudnnGetConvolutionNdForwardOutputDim(conv_desc : ConvolutionDescriptorT, input_tensor_desc : TensorDescriptorT, filter_desc : FilterDescriptorT, nb_dims : LibC::Int, tensor_ouput_dim_a : LibC::Int*) : StatusT
  fun destroy_convolution_descriptor = cudnnDestroyConvolutionDescriptor(conv_desc : ConvolutionDescriptorT) : StatusT
  fun get_convolution_forward_algorithm_max_count = cudnnGetConvolutionForwardAlgorithmMaxCount(handle : HandleT, count : LibC::Int*) : StatusT
  fun find_convolution_forward_algorithm = cudnnFindConvolutionForwardAlgorithm(handle : HandleT, x_desc : TensorDescriptorT, w_desc : FilterDescriptorT, conv_desc : ConvolutionDescriptorT, y_desc : TensorDescriptorT, requested_algo_count : LibC::Int, returned_algo_count : LibC::Int*, perf_results : ConvolutionFwdAlgoPerfT*) : StatusT
  fun find_convolution_forward_algorithm_ex = cudnnFindConvolutionForwardAlgorithmEx(handle : HandleT, x_desc : TensorDescriptorT, x : Void*, w_desc : FilterDescriptorT, w : Void*, conv_desc : ConvolutionDescriptorT, y_desc : TensorDescriptorT, y : Void*, requested_algo_count : LibC::Int, returned_algo_count : LibC::Int*, perf_results : ConvolutionFwdAlgoPerfT*, work_space : Void*, work_space_size_in_bytes : LibC::Int) : StatusT
  fun get_convolution_forward_algorithm = cudnnGetConvolutionForwardAlgorithm(handle : HandleT, x_desc : TensorDescriptorT, w_desc : FilterDescriptorT, conv_desc : ConvolutionDescriptorT, y_desc : TensorDescriptorT, preference : ConvolutionFwdPreferenceT, memory_limit_in_bytes : LibC::Int, algo : ConvolutionFwdAlgoT*) : StatusT
  fun get_convolution_forward_algorithm_v7 = cudnnGetConvolutionForwardAlgorithm_v7(handle : HandleT, src_desc : TensorDescriptorT, filter_desc : FilterDescriptorT, conv_desc : ConvolutionDescriptorT, dest_desc : TensorDescriptorT, requested_algo_count : LibC::Int, returned_algo_count : LibC::Int*, perf_results : ConvolutionFwdAlgoPerfT*) : StatusT
  fun get_convolution_forward_workspace_size = cudnnGetConvolutionForwardWorkspaceSize(handle : HandleT, x_desc : TensorDescriptorT, w_desc : FilterDescriptorT, conv_desc : ConvolutionDescriptorT, y_desc : TensorDescriptorT, algo : ConvolutionFwdAlgoT, size_in_bytes : LibC::Int*) : StatusT
  fun convolution_forward = cudnnConvolutionForward(handle : HandleT, alpha : Void*, x_desc : TensorDescriptorT, x : Void*, w_desc : FilterDescriptorT, w : Void*, conv_desc : ConvolutionDescriptorT, algo : ConvolutionFwdAlgoT, work_space : Void*, work_space_size_in_bytes : LibC::Int, beta : Void*, y_desc : TensorDescriptorT, y : Void*) : StatusT
  fun convolution_bias_activation_forward = cudnnConvolutionBiasActivationForward(handle : HandleT, alpha1 : Void*, x_desc : TensorDescriptorT, x : Void*, w_desc : FilterDescriptorT, w : Void*, conv_desc : ConvolutionDescriptorT, algo : ConvolutionFwdAlgoT, work_space : Void*, work_space_size_in_bytes : LibC::Int, alpha2 : Void*, z_desc : TensorDescriptorT, z : Void*, bias_desc : TensorDescriptorT, bias : Void*, activation_desc : ActivationDescriptorT, y_desc : TensorDescriptorT, y : Void*) : StatusT
  fun convolution_backward_bias = cudnnConvolutionBackwardBias(handle : HandleT, alpha : Void*, dy_desc : TensorDescriptorT, dy : Void*, beta : Void*, db_desc : TensorDescriptorT, db : Void*) : StatusT
  fun get_convolution_backward_filter_algorithm_max_count = cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(handle : HandleT, count : LibC::Int*) : StatusT
  fun find_convolution_backward_filter_algorithm = cudnnFindConvolutionBackwardFilterAlgorithm(handle : HandleT, x_desc : TensorDescriptorT, dy_desc : TensorDescriptorT, conv_desc : ConvolutionDescriptorT, dw_desc : FilterDescriptorT, requested_algo_count : LibC::Int, returned_algo_count : LibC::Int*, perf_results : ConvolutionBwdFilterAlgoPerfT*) : StatusT
  fun find_convolution_backward_filter_algorithm_ex = cudnnFindConvolutionBackwardFilterAlgorithmEx(handle : HandleT, x_desc : TensorDescriptorT, x : Void*, dy_desc : TensorDescriptorT, y : Void*, conv_desc : ConvolutionDescriptorT, dw_desc : FilterDescriptorT, dw : Void*, requested_algo_count : LibC::Int, returned_algo_count : LibC::Int*, perf_results : ConvolutionBwdFilterAlgoPerfT*, work_space : Void*, work_space_size_in_bytes : LibC::Int) : StatusT
  fun get_convolution_backward_filter_algorithm = cudnnGetConvolutionBackwardFilterAlgorithm(handle : HandleT, x_desc : TensorDescriptorT, dy_desc : TensorDescriptorT, conv_desc : ConvolutionDescriptorT, dw_desc : FilterDescriptorT, preference : ConvolutionBwdFilterPreferenceT, memory_limit_in_bytes : LibC::Int, algo : ConvolutionBwdFilterAlgoT*) : StatusT
  fun get_convolution_backward_filter_algorithm_v7 = cudnnGetConvolutionBackwardFilterAlgorithm_v7(handle : HandleT, src_desc : TensorDescriptorT, diff_desc : TensorDescriptorT, conv_desc : ConvolutionDescriptorT, grad_desc : FilterDescriptorT, requested_algo_count : LibC::Int, returned_algo_count : LibC::Int*, perf_results : ConvolutionBwdFilterAlgoPerfT*) : StatusT
  fun get_convolution_backward_filter_workspace_size = cudnnGetConvolutionBackwardFilterWorkspaceSize(handle : HandleT, x_desc : TensorDescriptorT, dy_desc : TensorDescriptorT, conv_desc : ConvolutionDescriptorT, grad_desc : FilterDescriptorT, algo : ConvolutionBwdFilterAlgoT, size_in_bytes : LibC::Int*) : StatusT
  fun convolution_backward_filter = cudnnConvolutionBackwardFilter(handle : HandleT, alpha : Void*, x_desc : TensorDescriptorT, x : Void*, dy_desc : TensorDescriptorT, dy : Void*, conv_desc : ConvolutionDescriptorT, algo : ConvolutionBwdFilterAlgoT, work_space : Void*, work_space_size_in_bytes : LibC::Int, beta : Void*, dw_desc : FilterDescriptorT, dw : Void*) : StatusT
  fun get_convolution_backward_data_algorithm_max_count = cudnnGetConvolutionBackwardDataAlgorithmMaxCount(handle : HandleT, count : LibC::Int*) : StatusT
  fun find_convolution_backward_data_algorithm = cudnnFindConvolutionBackwardDataAlgorithm(handle : HandleT, w_desc : FilterDescriptorT, dy_desc : TensorDescriptorT, conv_desc : ConvolutionDescriptorT, dx_desc : TensorDescriptorT, requested_algo_count : LibC::Int, returned_algo_count : LibC::Int*, perf_results : ConvolutionBwdDataAlgoPerfT*) : StatusT
  fun find_convolution_backward_data_algorithm_ex = cudnnFindConvolutionBackwardDataAlgorithmEx(handle : HandleT, w_desc : FilterDescriptorT, w : Void*, dy_desc : TensorDescriptorT, dy : Void*, conv_desc : ConvolutionDescriptorT, dx_desc : TensorDescriptorT, dx : Void*, requested_algo_count : LibC::Int, returned_algo_count : LibC::Int*, perf_results : ConvolutionBwdDataAlgoPerfT*, work_space : Void*, work_space_size_in_bytes : LibC::Int) : StatusT
  fun get_convolution_backward_data_algorithm = cudnnGetConvolutionBackwardDataAlgorithm(handle : HandleT, w_desc : FilterDescriptorT, dy_desc : TensorDescriptorT, conv_desc : ConvolutionDescriptorT, dx_desc : TensorDescriptorT, preference : ConvolutionBwdDataPreferenceT, memory_limit_in_bytes : LibC::Int, algo : ConvolutionBwdDataAlgoT*) : StatusT
  fun get_convolution_backward_data_algorithm_v7 = cudnnGetConvolutionBackwardDataAlgorithm_v7(handle : HandleT, filter_desc : FilterDescriptorT, diff_desc : TensorDescriptorT, conv_desc : ConvolutionDescriptorT, grad_desc : TensorDescriptorT, requested_algo_count : LibC::Int, returned_algo_count : LibC::Int*, perf_results : ConvolutionBwdDataAlgoPerfT*) : StatusT
  fun get_convolution_backward_data_workspace_size = cudnnGetConvolutionBackwardDataWorkspaceSize(handle : HandleT, w_desc : FilterDescriptorT, dy_desc : TensorDescriptorT, conv_desc : ConvolutionDescriptorT, dx_desc : TensorDescriptorT, algo : ConvolutionBwdDataAlgoT, size_in_bytes : LibC::Int*) : StatusT
  fun convolution_backward_data = cudnnConvolutionBackwardData(handle : HandleT, alpha : Void*, w_desc : FilterDescriptorT, w : Void*, dy_desc : TensorDescriptorT, dy : Void*, conv_desc : ConvolutionDescriptorT, algo : ConvolutionBwdDataAlgoT, work_space : Void*, work_space_size_in_bytes : LibC::Int, beta : Void*, dx_desc : TensorDescriptorT, dx : Void*) : StatusT
  fun im2_col = cudnnIm2Col(handle : HandleT, x_desc : TensorDescriptorT, x : Void*, w_desc : FilterDescriptorT, conv_desc : ConvolutionDescriptorT, col_buffer : Void*) : StatusT
  fun softmax_forward = cudnnSoftmaxForward(handle : HandleT, algo : SoftmaxAlgorithmT, mode : SoftmaxModeT, alpha : Void*, x_desc : TensorDescriptorT, x : Void*, beta : Void*, y_desc : TensorDescriptorT, y : Void*) : StatusT
  fun softmax_backward = cudnnSoftmaxBackward(handle : HandleT, algo : SoftmaxAlgorithmT, mode : SoftmaxModeT, alpha : Void*, y_desc : TensorDescriptorT, y : Void*, dy_desc : TensorDescriptorT, dy : Void*, beta : Void*, dx_desc : TensorDescriptorT, dx : Void*) : StatusT
  fun create_pooling_descriptor = cudnnCreatePoolingDescriptor(pooling_desc : PoolingDescriptorT*) : StatusT
  fun set_pooling2d_descriptor = cudnnSetPooling2dDescriptor(pooling_desc : PoolingDescriptorT, mode : PoolingModeT, maxpooling_nan_opt : NanPropagationT, window_height : LibC::Int, window_width : LibC::Int, vertical_padding : LibC::Int, horizontal_padding : LibC::Int, vertical_stride : LibC::Int, horizontal_stride : LibC::Int) : StatusT
  fun get_pooling2d_descriptor = cudnnGetPooling2dDescriptor(pooling_desc : PoolingDescriptorT, mode : PoolingModeT*, maxpooling_nan_opt : NanPropagationT*, window_height : LibC::Int*, window_width : LibC::Int*, vertical_padding : LibC::Int*, horizontal_padding : LibC::Int*, vertical_stride : LibC::Int*, horizontal_stride : LibC::Int*) : StatusT
  fun set_pooling_nd_descriptor = cudnnSetPoolingNdDescriptor(pooling_desc : PoolingDescriptorT, mode : PoolingModeT, maxpooling_nan_opt : NanPropagationT, nb_dims : LibC::Int, window_dim_a : LibC::Int*, padding_a : LibC::Int*, stride_a : LibC::Int*) : StatusT
  fun get_pooling_nd_descriptor = cudnnGetPoolingNdDescriptor(pooling_desc : PoolingDescriptorT, nb_dims_requested : LibC::Int, mode : PoolingModeT*, maxpooling_nan_opt : NanPropagationT*, nb_dims : LibC::Int*, window_dim_a : LibC::Int*, padding_a : LibC::Int*, stride_a : LibC::Int*) : StatusT
  fun get_pooling_nd_forward_output_dim = cudnnGetPoolingNdForwardOutputDim(pooling_desc : PoolingDescriptorT, input_tensor_desc : TensorDescriptorT, nb_dims : LibC::Int, output_tensor_dim_a : LibC::Int*) : StatusT
  fun get_pooling2d_forward_output_dim = cudnnGetPooling2dForwardOutputDim(pooling_desc : PoolingDescriptorT, input_tensor_desc : TensorDescriptorT, n : LibC::Int*, c : LibC::Int*, h : LibC::Int*, w : LibC::Int*) : StatusT
  fun destroy_pooling_descriptor = cudnnDestroyPoolingDescriptor(pooling_desc : PoolingDescriptorT) : StatusT
  fun pooling_forward = cudnnPoolingForward(handle : HandleT, pooling_desc : PoolingDescriptorT, alpha : Void*, x_desc : TensorDescriptorT, x : Void*, beta : Void*, y_desc : TensorDescriptorT, y : Void*) : StatusT
  fun pooling_backward = cudnnPoolingBackward(handle : HandleT, pooling_desc : PoolingDescriptorT, alpha : Void*, y_desc : TensorDescriptorT, y : Void*, dy_desc : TensorDescriptorT, dy : Void*, x_desc : TensorDescriptorT, x : Void*, beta : Void*, dx_desc : TensorDescriptorT, dx : Void*) : StatusT
  fun create_activation_descriptor = cudnnCreateActivationDescriptor(activation_desc : ActivationDescriptorT*) : StatusT
  fun set_activation_descriptor = cudnnSetActivationDescriptor(activation_desc : ActivationDescriptorT, mode : ActivationModeT, relu_nan_opt : NanPropagationT, coef : LibC::Double) : StatusT
  fun get_activation_descriptor = cudnnGetActivationDescriptor(activation_desc : ActivationDescriptorT, mode : ActivationModeT*, relu_nan_opt : NanPropagationT*, coef : LibC::Double*) : StatusT
  fun destroy_activation_descriptor = cudnnDestroyActivationDescriptor(activation_desc : ActivationDescriptorT) : StatusT
  fun activation_forward = cudnnActivationForward(handle : HandleT, activation_desc : ActivationDescriptorT, alpha : Void*, x_desc : TensorDescriptorT, x : Void*, beta : Void*, y_desc : TensorDescriptorT, y : Void*) : StatusT
  fun activation_backward = cudnnActivationBackward(handle : HandleT, activation_desc : ActivationDescriptorT, alpha : Void*, y_desc : TensorDescriptorT, y : Void*, dy_desc : TensorDescriptorT, dy : Void*, x_desc : TensorDescriptorT, x : Void*, beta : Void*, dx_desc : TensorDescriptorT, dx : Void*) : StatusT
  fun create_lrn_descriptor = cudnnCreateLRNDescriptor(norm_desc : LrnDescriptorT*) : StatusT
  fun set_lrn_descriptor = cudnnSetLRNDescriptor(norm_desc : LrnDescriptorT, lrn_n : LibC::UInt, lrn_alpha : LibC::Double, lrn_beta : LibC::Double, lrn_k : LibC::Double) : StatusT
  fun get_lrn_descriptor = cudnnGetLRNDescriptor(norm_desc : LrnDescriptorT, lrn_n : LibC::UInt*, lrn_alpha : LibC::Double*, lrn_beta : LibC::Double*, lrn_k : LibC::Double*) : StatusT
  fun destroy_lrn_descriptor = cudnnDestroyLRNDescriptor(lrn_desc : LrnDescriptorT) : StatusT
  fun lrn_cross_channel_forward = cudnnLRNCrossChannelForward(handle : HandleT, norm_desc : LrnDescriptorT, lrn_mode : LrnModeT, alpha : Void*, x_desc : TensorDescriptorT, x : Void*, beta : Void*, y_desc : TensorDescriptorT, y : Void*) : StatusT
  fun lrn_cross_channel_backward = cudnnLRNCrossChannelBackward(handle : HandleT, norm_desc : LrnDescriptorT, lrn_mode : LrnModeT, alpha : Void*, y_desc : TensorDescriptorT, y : Void*, dy_desc : TensorDescriptorT, dy : Void*, x_desc : TensorDescriptorT, x : Void*, beta : Void*, dx_desc : TensorDescriptorT, dx : Void*) : StatusT
  fun divisive_normalization_forward = cudnnDivisiveNormalizationForward(handle : HandleT, norm_desc : LrnDescriptorT, mode : DivNormModeT, alpha : Void*, x_desc : TensorDescriptorT, x : Void*, means : Void*, temp : Void*, temp2 : Void*, beta : Void*, y_desc : TensorDescriptorT, y : Void*) : StatusT
  fun divisive_normalization_backward = cudnnDivisiveNormalizationBackward(handle : HandleT, norm_desc : LrnDescriptorT, mode : DivNormModeT, alpha : Void*, x_desc : TensorDescriptorT, x : Void*, means : Void*, dy : Void*, temp : Void*, temp2 : Void*, beta : Void*, d_xd_means_desc : TensorDescriptorT, dx : Void*, d_means : Void*) : StatusT
  fun derive_bn_tensor_descriptor = cudnnDeriveBNTensorDescriptor(derived_bn_desc : TensorDescriptorT, x_desc : TensorDescriptorT, mode : BatchNormModeT) : StatusT
  fun batch_normalization_forward_training = cudnnBatchNormalizationForwardTraining(handle : HandleT, mode : BatchNormModeT, alpha : Void*, beta : Void*, x_desc : TensorDescriptorT, x : Void*, y_desc : TensorDescriptorT, y : Void*, bn_scale_bias_mean_var_desc : TensorDescriptorT, bn_scale : Void*, bn_bias : Void*, exponential_average_factor : LibC::Double, result_running_mean : Void*, result_running_variance : Void*, epsilon : LibC::Double, result_save_mean : Void*, result_save_inv_variance : Void*) : StatusT
  fun batch_normalization_forward_inference = cudnnBatchNormalizationForwardInference(handle : HandleT, mode : BatchNormModeT, alpha : Void*, beta : Void*, x_desc : TensorDescriptorT, x : Void*, y_desc : TensorDescriptorT, y : Void*, bn_scale_bias_mean_var_desc : TensorDescriptorT, bn_scale : Void*, bn_bias : Void*, estimated_mean : Void*, estimated_variance : Void*, epsilon : LibC::Double) : StatusT
  fun batch_normalization_backward = cudnnBatchNormalizationBackward(handle : HandleT, mode : BatchNormModeT, alpha_data_diff : Void*, beta_data_diff : Void*, alpha_param_diff : Void*, beta_param_diff : Void*, x_desc : TensorDescriptorT, x : Void*, dy_desc : TensorDescriptorT, dy : Void*, dx_desc : TensorDescriptorT, dx : Void*, d_bn_scale_bias_desc : TensorDescriptorT, bn_scale : Void*, d_bn_scale_result : Void*, d_bn_bias_result : Void*, epsilon : LibC::Double, saved_mean : Void*, saved_inv_variance : Void*) : StatusT
  fun create_spatial_transformer_descriptor = cudnnCreateSpatialTransformerDescriptor(st_desc : SpatialTransformerDescriptorT*) : StatusT
  fun set_spatial_transformer_nd_descriptor = cudnnSetSpatialTransformerNdDescriptor(st_desc : SpatialTransformerDescriptorT, sampler_type : SamplerTypeT, data_type : DataTypeT, nb_dims : LibC::Int, dim_a : LibC::Int*) : StatusT
  fun destroy_spatial_transformer_descriptor = cudnnDestroySpatialTransformerDescriptor(st_desc : SpatialTransformerDescriptorT) : StatusT
  fun spatial_tf_grid_generator_forward = cudnnSpatialTfGridGeneratorForward(handle : HandleT, st_desc : SpatialTransformerDescriptorT, theta : Void*, grid : Void*) : StatusT
  fun spatial_tf_grid_generator_backward = cudnnSpatialTfGridGeneratorBackward(handle : HandleT, st_desc : SpatialTransformerDescriptorT, dgrid : Void*, dtheta : Void*) : StatusT
  fun spatial_tf_sampler_forward = cudnnSpatialTfSamplerForward(handle : HandleT, st_desc : SpatialTransformerDescriptorT, alpha : Void*, x_desc : TensorDescriptorT, x : Void*, grid : Void*, beta : Void*, y_desc : TensorDescriptorT, y : Void*) : StatusT
  fun spatial_tf_sampler_backward = cudnnSpatialTfSamplerBackward(handle : HandleT, st_desc : SpatialTransformerDescriptorT, alpha : Void*, x_desc : TensorDescriptorT, x : Void*, beta : Void*, dx_desc : TensorDescriptorT, dx : Void*, alpha_dgrid : Void*, dy_desc : TensorDescriptorT, dy : Void*, grid : Void*, beta_dgrid : Void*, dgrid : Void*) : StatusT
  fun create_dropout_descriptor = cudnnCreateDropoutDescriptor(dropout_desc : DropoutDescriptorT*) : StatusT
  fun destroy_dropout_descriptor = cudnnDestroyDropoutDescriptor(dropout_desc : DropoutDescriptorT) : StatusT
  fun dropout_get_states_size = cudnnDropoutGetStatesSize(handle : HandleT, size_in_bytes : LibC::Int*) : StatusT
  fun dropout_get_reserve_space_size = cudnnDropoutGetReserveSpaceSize(xdesc : TensorDescriptorT, size_in_bytes : LibC::Int*) : StatusT
  fun set_dropout_descriptor = cudnnSetDropoutDescriptor(dropout_desc : DropoutDescriptorT, handle : HandleT, dropout : LibC::Float, states : Void*, state_size_in_bytes : LibC::Int, seed : LibC::ULongLong) : StatusT
  fun restore_dropout_descriptor = cudnnRestoreDropoutDescriptor(dropout_desc : DropoutDescriptorT, handle : HandleT, dropout : LibC::Float, states : Void*, state_size_in_bytes : LibC::Int, seed : LibC::ULongLong) : StatusT
  fun get_dropout_descriptor = cudnnGetDropoutDescriptor(dropout_desc : DropoutDescriptorT, handle : HandleT, dropout : LibC::Float*, states : Void**, seed : LibC::ULongLong*) : StatusT
  fun dropout_forward = cudnnDropoutForward(handle : HandleT, dropout_desc : DropoutDescriptorT, xdesc : TensorDescriptorT, x : Void*, ydesc : TensorDescriptorT, y : Void*, reserve_space : Void*, reserve_space_size_in_bytes : LibC::Int) : StatusT
  fun dropout_backward = cudnnDropoutBackward(handle : HandleT, dropout_desc : DropoutDescriptorT, dydesc : TensorDescriptorT, dy : Void*, dxdesc : TensorDescriptorT, dx : Void*, reserve_space : Void*, reserve_space_size_in_bytes : LibC::Int) : StatusT
  fun create_rnn_descriptor = cudnnCreateRNNDescriptor(rnn_desc : RnnDescriptorT*) : StatusT
  fun destroy_rnn_descriptor = cudnnDestroyRNNDescriptor(rnn_desc : RnnDescriptorT) : StatusT
  fun get_rnn_forward_inference_algorithm_max_count = cudnnGetRNNForwardInferenceAlgorithmMaxCount(handle : HandleT, rnn_desc : RnnDescriptorT, count : LibC::Int*) : StatusT
  fun find_rnn_forward_inference_algorithm_ex = cudnnFindRNNForwardInferenceAlgorithmEx(handle : HandleT, rnn_desc : RnnDescriptorT, seq_length : LibC::Int, x_desc : TensorDescriptorT*, x : Void*, hx_desc : TensorDescriptorT, hx : Void*, cx_desc : TensorDescriptorT, cx : Void*, w_desc : FilterDescriptorT, w : Void*, y_desc : TensorDescriptorT*, y : Void*, hy_desc : TensorDescriptorT, hy : Void*, cy_desc : TensorDescriptorT, cy : Void*, find_intensity : LibC::Float, requested_algo_count : LibC::Int, returned_algo_count : LibC::Int*, perf_results : AlgorithmPerformanceT*, workspace : Void*, work_space_size_in_bytes : LibC::Int) : StatusT
  fun get_rnn_forward_training_algorithm_max_count = cudnnGetRNNForwardTrainingAlgorithmMaxCount(handle : HandleT, rnn_desc : RnnDescriptorT, count : LibC::Int*) : StatusT
  fun find_rnn_forward_training_algorithm_ex = cudnnFindRNNForwardTrainingAlgorithmEx(handle : HandleT, rnn_desc : RnnDescriptorT, seq_length : LibC::Int, x_desc : TensorDescriptorT*, x : Void*, hx_desc : TensorDescriptorT, hx : Void*, cx_desc : TensorDescriptorT, cx : Void*, w_desc : FilterDescriptorT, w : Void*, y_desc : TensorDescriptorT*, y : Void*, hy_desc : TensorDescriptorT, hy : Void*, cy_desc : TensorDescriptorT, cy : Void*, find_intensity : LibC::Float, requested_algo_count : LibC::Int, returned_algo_count : LibC::Int*, perf_results : AlgorithmPerformanceT*, workspace : Void*, work_space_size_in_bytes : LibC::Int, reserve_space : Void*, reserve_space_size_in_bytes : LibC::Int) : StatusT
  fun get_rnn_backward_data_algorithm_max_count = cudnnGetRNNBackwardDataAlgorithmMaxCount(handle : HandleT, rnn_desc : RnnDescriptorT, count : LibC::Int*) : StatusT
  fun find_rnn_backward_data_algorithm_ex = cudnnFindRNNBackwardDataAlgorithmEx(handle : HandleT, rnn_desc : RnnDescriptorT, seq_length : LibC::Int, y_desc : TensorDescriptorT*, y : Void*, dy_desc : TensorDescriptorT*, dy : Void*, dhy_desc : TensorDescriptorT, dhy : Void*, dcy_desc : TensorDescriptorT, dcy : Void*, w_desc : FilterDescriptorT, w : Void*, hx_desc : TensorDescriptorT, hx : Void*, cx_desc : TensorDescriptorT, cx : Void*, dx_desc : TensorDescriptorT*, dx : Void*, dhx_desc : TensorDescriptorT, dhx : Void*, dcx_desc : TensorDescriptorT, dcx : Void*, find_intensity : LibC::Float, requested_algo_count : LibC::Int, returned_algo_count : LibC::Int*, perf_results : AlgorithmPerformanceT*, workspace : Void*, work_space_size_in_bytes : LibC::Int, reserve_space : Void*, reserve_space_size_in_bytes : LibC::Int) : StatusT
  fun get_rnn_backward_weights_algorithm_max_count = cudnnGetRNNBackwardWeightsAlgorithmMaxCount(handle : HandleT, rnn_desc : RnnDescriptorT, count : LibC::Int*) : StatusT
  fun find_rnn_backward_weights_algorithm_ex = cudnnFindRNNBackwardWeightsAlgorithmEx(handle : HandleT, rnn_desc : RnnDescriptorT, seq_length : LibC::Int, x_desc : TensorDescriptorT*, x : Void*, hx_desc : TensorDescriptorT, hx : Void*, y_desc : TensorDescriptorT*, y : Void*, find_intensity : LibC::Float, requested_algo_count : LibC::Int, returned_algo_count : LibC::Int*, perf_results : AlgorithmPerformanceT*, workspace : Void*, work_space_size_in_bytes : LibC::Int, dw_desc : FilterDescriptorT, dw : Void*, reserve_space : Void*, reserve_space_size_in_bytes : LibC::Int) : StatusT
  fun create_persistent_rnn_plan = cudnnCreatePersistentRNNPlan(rnn_desc : RnnDescriptorT, minibatch : LibC::Int, data_type : DataTypeT, plan : PersistentRnnPlanT*) : StatusT
  fun set_persistent_rnn_plan = cudnnSetPersistentRNNPlan(rnn_desc : RnnDescriptorT, plan : PersistentRnnPlanT) : StatusT
  fun destroy_persistent_rnn_plan = cudnnDestroyPersistentRNNPlan(plan : PersistentRnnPlanT) : StatusT
  fun set_rnn_descriptor = cudnnSetRNNDescriptor(handle : HandleT, rnn_desc : RnnDescriptorT, hidden_size : LibC::Int, num_layers : LibC::Int, dropout_desc : DropoutDescriptorT, input_mode : RnnInputModeT, direction : DirectionModeT, mode : RnnModeT, algo : RnnAlgoT, data_type : DataTypeT) : StatusT
  fun set_rnn_projection_layers = cudnnSetRNNProjectionLayers(handle : HandleT, rnn_desc : RnnDescriptorT, rec_proj_size : LibC::Int, out_proj_size : LibC::Int) : StatusT
  fun get_rnn_projection_layers = cudnnGetRNNProjectionLayers(handle : HandleT, rnn_desc : RnnDescriptorT, rec_proj_size : LibC::Int*, out_proj_size : LibC::Int*) : StatusT
  fun set_rnn_algorithm_descriptor = cudnnSetRNNAlgorithmDescriptor(handle : HandleT, rnn_desc : RnnDescriptorT, algo_desc : AlgorithmDescriptorT) : StatusT
  fun get_rnn_descriptor = cudnnGetRNNDescriptor(handle : HandleT, rnn_desc : RnnDescriptorT, hidden_size : LibC::Int*, num_layers : LibC::Int*, dropout_desc : DropoutDescriptorT*, input_mode : RnnInputModeT*, direction : DirectionModeT*, mode : RnnModeT*, algo : RnnAlgoT*, data_type : DataTypeT*) : StatusT
  fun set_rnn_matrix_math_type = cudnnSetRNNMatrixMathType(rnn_desc : RnnDescriptorT, m_type : MathTypeT) : StatusT
  fun get_rnn_matrix_math_type = cudnnGetRNNMatrixMathType(rnn_desc : RnnDescriptorT, m_type : MathTypeT*) : StatusT
  fun get_rnn_workspace_size = cudnnGetRNNWorkspaceSize(handle : HandleT, rnn_desc : RnnDescriptorT, seq_length : LibC::Int, x_desc : TensorDescriptorT*, size_in_bytes : LibC::Int*) : StatusT
  fun get_rnn_training_reserve_size = cudnnGetRNNTrainingReserveSize(handle : HandleT, rnn_desc : RnnDescriptorT, seq_length : LibC::Int, x_desc : TensorDescriptorT*, size_in_bytes : LibC::Int*) : StatusT
  fun get_rnn_params_size = cudnnGetRNNParamsSize(handle : HandleT, rnn_desc : RnnDescriptorT, x_desc : TensorDescriptorT, size_in_bytes : LibC::Int*, data_type : DataTypeT) : StatusT
  fun get_rnn_lin_layer_matrix_params = cudnnGetRNNLinLayerMatrixParams(handle : HandleT, rnn_desc : RnnDescriptorT, pseudo_layer : LibC::Int, x_desc : TensorDescriptorT, w_desc : FilterDescriptorT, w : Void*, lin_layer_id : LibC::Int, lin_layer_mat_desc : FilterDescriptorT, lin_layer_mat : Void**) : StatusT
  fun get_rnn_lin_layer_bias_params = cudnnGetRNNLinLayerBiasParams(handle : HandleT, rnn_desc : RnnDescriptorT, pseudo_layer : LibC::Int, x_desc : TensorDescriptorT, w_desc : FilterDescriptorT, w : Void*, lin_layer_id : LibC::Int, lin_layer_bias_desc : FilterDescriptorT, lin_layer_bias : Void**) : StatusT
  fun rnn_forward_inference = cudnnRNNForwardInference(handle : HandleT, rnn_desc : RnnDescriptorT, seq_length : LibC::Int, x_desc : TensorDescriptorT*, x : Void*, hx_desc : TensorDescriptorT, hx : Void*, cx_desc : TensorDescriptorT, cx : Void*, w_desc : FilterDescriptorT, w : Void*, y_desc : TensorDescriptorT*, y : Void*, hy_desc : TensorDescriptorT, hy : Void*, cy_desc : TensorDescriptorT, cy : Void*, workspace : Void*, work_space_size_in_bytes : LibC::Int) : StatusT
  fun rnn_forward_training = cudnnRNNForwardTraining(handle : HandleT, rnn_desc : RnnDescriptorT, seq_length : LibC::Int, x_desc : TensorDescriptorT*, x : Void*, hx_desc : TensorDescriptorT, hx : Void*, cx_desc : TensorDescriptorT, cx : Void*, w_desc : FilterDescriptorT, w : Void*, y_desc : TensorDescriptorT*, y : Void*, hy_desc : TensorDescriptorT, hy : Void*, cy_desc : TensorDescriptorT, cy : Void*, workspace : Void*, work_space_size_in_bytes : LibC::Int, reserve_space : Void*, reserve_space_size_in_bytes : LibC::Int) : StatusT
  fun rnn_backward_data = cudnnRNNBackwardData(handle : HandleT, rnn_desc : RnnDescriptorT, seq_length : LibC::Int, y_desc : TensorDescriptorT*, y : Void*, dy_desc : TensorDescriptorT*, dy : Void*, dhy_desc : TensorDescriptorT, dhy : Void*, dcy_desc : TensorDescriptorT, dcy : Void*, w_desc : FilterDescriptorT, w : Void*, hx_desc : TensorDescriptorT, hx : Void*, cx_desc : TensorDescriptorT, cx : Void*, dx_desc : TensorDescriptorT*, dx : Void*, dhx_desc : TensorDescriptorT, dhx : Void*, dcx_desc : TensorDescriptorT, dcx : Void*, workspace : Void*, work_space_size_in_bytes : LibC::Int, reserve_space : Void*, reserve_space_size_in_bytes : LibC::Int) : StatusT
  fun rnn_backward_weights = cudnnRNNBackwardWeights(handle : HandleT, rnn_desc : RnnDescriptorT, seq_length : LibC::Int, x_desc : TensorDescriptorT*, x : Void*, hx_desc : TensorDescriptorT, hx : Void*, y_desc : TensorDescriptorT*, y : Void*, workspace : Void*, work_space_size_in_bytes : LibC::Int, dw_desc : FilterDescriptorT, dw : Void*, reserve_space : Void*, reserve_space_size_in_bytes : LibC::Int) : StatusT
  fun create_ctc_loss_descriptor = cudnnCreateCTCLossDescriptor(ctc_loss_desc : CtcLossDescriptorT*) : StatusT
  fun set_ctc_loss_descriptor = cudnnSetCTCLossDescriptor(ctc_loss_desc : CtcLossDescriptorT, comp_type : DataTypeT) : StatusT
  fun get_ctc_loss_descriptor = cudnnGetCTCLossDescriptor(ctc_loss_desc : CtcLossDescriptorT, comp_type : DataTypeT*) : StatusT
  fun destroy_ctc_loss_descriptor = cudnnDestroyCTCLossDescriptor(ctc_loss_desc : CtcLossDescriptorT) : StatusT
  fun ctc_loss = cudnnCTCLoss(handle : HandleT, probs_desc : TensorDescriptorT, probs : Void*, labels : LibC::Int*, label_lengths : LibC::Int*, input_lengths : LibC::Int*, costs : Void*, gradients_desc : TensorDescriptorT, gradients : Void*, algo : CtcLossAlgoT, ctc_loss_desc : CtcLossDescriptorT, workspace : Void*, work_space_size_in_bytes : LibC::Int) : StatusT
  fun get_ctc_loss_workspace_size = cudnnGetCTCLossWorkspaceSize(handle : HandleT, probs_desc : TensorDescriptorT, gradients_desc : TensorDescriptorT, labels : LibC::Int*, label_lengths : LibC::Int*, input_lengths : LibC::Int*, algo : CtcLossAlgoT, ctc_loss_desc : CtcLossDescriptorT, size_in_bytes : LibC::Int*) : StatusT
  fun create_algorithm_descriptor = cudnnCreateAlgorithmDescriptor(algo_desc : AlgorithmDescriptorT*) : StatusT
  fun set_algorithm_descriptor = cudnnSetAlgorithmDescriptor(algo_desc : AlgorithmDescriptorT, algorithm : AlgorithmT) : StatusT
  fun get_algorithm_descriptor = cudnnGetAlgorithmDescriptor(algo_desc : AlgorithmDescriptorT, algorithm : AlgorithmT*) : StatusT
  fun copy_algorithm_descriptor = cudnnCopyAlgorithmDescriptor(src : AlgorithmDescriptorT, dest : AlgorithmDescriptorT) : StatusT
  fun destroy_algorithm_descriptor = cudnnDestroyAlgorithmDescriptor(algo_desc : AlgorithmDescriptorT) : StatusT
  fun create_algorithm_performance = cudnnCreateAlgorithmPerformance(algo_perf : AlgorithmPerformanceT*, number_to_create : LibC::Int) : StatusT
  fun set_algorithm_performance = cudnnSetAlgorithmPerformance(algo_perf : AlgorithmPerformanceT, algo_desc : AlgorithmDescriptorT, status : StatusT, time : LibC::Float, memory : LibC::Int) : StatusT
  fun get_algorithm_performance = cudnnGetAlgorithmPerformance(algo_perf : AlgorithmPerformanceT, algo_desc : AlgorithmDescriptorT*, status : StatusT*, time : LibC::Float*, memory : LibC::Int*) : StatusT
  fun destroy_algorithm_performance = cudnnDestroyAlgorithmPerformance(algo_perf : AlgorithmPerformanceT*, number_to_destroy : LibC::Int) : StatusT
  fun get_algorithm_space_size = cudnnGetAlgorithmSpaceSize(handle : HandleT, algo_desc : AlgorithmDescriptorT, algo_space_size_in_bytes : LibC::Int*) : StatusT
  fun save_algorithm = cudnnSaveAlgorithm(handle : HandleT, algo_desc : AlgorithmDescriptorT, algo_space : Void*, algo_space_size_in_bytes : LibC::Int) : StatusT
  fun restore_algorithm = cudnnRestoreAlgorithm(handle : HandleT, algo_space : Void*, algo_space_size_in_bytes : LibC::Int, algo_desc : AlgorithmDescriptorT) : StatusT
  fun set_callback = cudnnSetCallback(mask : LibC::UInt, udata : Void*, fptr : CallbackT) : StatusT
  fun get_callback = cudnnGetCallback(mask : LibC::UInt*, udata : Void**, fptr : CallbackT*) : StatusT
  fun set_rnn_descriptor_v6 = cudnnSetRNNDescriptor_v6(handle : HandleT, rnn_desc : RnnDescriptorT, hidden_size : LibC::Int, num_layers : LibC::Int, dropout_desc : DropoutDescriptorT, input_mode : RnnInputModeT, direction : DirectionModeT, mode : RnnModeT, algo : RnnAlgoT, data_type : DataTypeT) : StatusT
  fun set_rnn_descriptor_v5 = cudnnSetRNNDescriptor_v5(rnn_desc : RnnDescriptorT, hidden_size : LibC::Int, num_layers : LibC::Int, dropout_desc : DropoutDescriptorT, input_mode : RnnInputModeT, direction : DirectionModeT, mode : RnnModeT, data_type : DataTypeT) : StatusT
end
