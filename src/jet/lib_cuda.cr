@[Link("cudart")]
lib LibCUDA
  HOSTALLOCDEFAULT                       =    0
  HOSTALLOCPORTABLE                      =    1
  HOSTALLOCMAPPED                        =    2
  HOSTALLOCWRITECOMBINED                 =    4
  HOSTREGISTERDEFAULT                    =    0
  HOSTREGISTERPORTABLE                   =    1
  HOSTREGISTERMAPPED                     =    2
  HOSTREGISTERIOMEMORY                   =    4
  PEERACCESSDEFAULT                      =    0
  STREAMDEFAULT                          =    0
  STREAMNONBLOCKING                      =    1
  EVENTDEFAULT                           =    0
  EVENTBLOCKINGSYNC                      =    1
  EVENTDISABLETIMING                     =    2
  EVENTINTERPROCESS                      =    4
  DEVICESCHEDULEAUTO                     =    0
  DEVICESCHEDULESPIN                     =    1
  DEVICESCHEDULEYIELD                    =    2
  DEVICESCHEDULEBLOCKINGSYNC             =    4
  DEVICEBLOCKINGSYNC                     =    4
  DEVICESCHEDULEMASK                     =    7
  DEVICEMAPHOST                          =    8
  DEVICELMEMRESIZETOMAX                  =   16
  DEVICEMASK                             =   31
  ARRAYDEFAULT                           =    0
  ARRAYLAYERED                           =    1
  ARRAYSURFACELOADSTORE                  =    2
  ARRAYCUBEMAP                           =    4
  ARRAYTEXTUREGATHER                     =    8
  IPCMEMLAZYENABLEPEERACCESS             =    1
  MEMATTACHGLOBAL                        =    1
  MEMATTACHHOST                          =    2
  MEMATTACHSINGLE                        =    4
  OCCUPANCYDEFAULT                       =    0
  OCCUPANCYDISABLECACHINGOVERRIDE        =    1
  COOPERATIVELAUNCHMULTIDEVICENOPRESYNC  =    1
  COOPERATIVELAUNCHMULTIDEVICENOPOSTSYNC =    2
  IPC_HANDLE_SIZE                        =   64
  SURFACETYPE1D                          =    1
  SURFACETYPE2D                          =    2
  SURFACETYPE3D                          =    3
  SURFACETYPECUBEMAP                     =   12
  SURFACETYPE1DLAYERED                   =  241
  SURFACETYPE2DLAYERED                   =  242
  SURFACETYPECUBEMAPLAYERED              =  252
  TEXTURETYPE1D                          =    1
  TEXTURETYPE2D                          =    2
  TEXTURETYPE3D                          =    3
  TEXTURETYPECUBEMAP                     =   12
  TEXTURETYPE1DLAYERED                   =  241
  TEXTURETYPE2DLAYERED                   =  242
  TEXTURETYPECUBEMAPLAYERED              =  252
  RT_VERSION                             = 9010

  alias Array = Void
  alias MipmappedArray = Void
  alias GraphicsResource = Void
  alias ErrorT = Error
  alias IpcMemHandleT = IpcMemHandleSt
  alias IpcEventHandleT = IpcEventHandleSt

  type ArrayT = Void*
  type MipmappedArrayT = Void*
  type StreamT = Void*
  type EventT = Void*

  enum ChannelFormatKind
    ChannelFormatKindSigned   = 0
    ChannelFormatKindUnsigned = 1
    ChannelFormatKindFloat    = 2
    ChannelFormatKindNone     = 3
  end

  enum MemcpyKind
    MemcpyHostToHost     = 0
    MemcpyHostToDevice   = 1
    MemcpyDeviceToHost   = 2
    MemcpyDeviceToDevice = 3
    MemcpyDefault        = 4
  end

  enum ResourceType
    ResourceTypeArray          = 0
    ResourceTypeMipmappedArray = 1
    ResourceTypeLinear         = 2
    ResourceTypePitch2D        = 3
  end

  enum ResourceViewFormat
    ResViewFormatNone                      =  0
    ResViewFormatUnsignedChar1             =  1
    ResViewFormatUnsignedChar2             =  2
    ResViewFormatUnsignedChar4             =  3
    ResViewFormatSignedChar1               =  4
    ResViewFormatSignedChar2               =  5
    ResViewFormatSignedChar4               =  6
    ResViewFormatUnsignedShort1            =  7
    ResViewFormatUnsignedShort2            =  8
    ResViewFormatUnsignedShort4            =  9
    ResViewFormatSignedShort1              = 10
    ResViewFormatSignedShort2              = 11
    ResViewFormatSignedShort4              = 12
    ResViewFormatUnsignedInt1              = 13
    ResViewFormatUnsignedInt2              = 14
    ResViewFormatUnsignedInt4              = 15
    ResViewFormatSignedInt1                = 16
    ResViewFormatSignedInt2                = 17
    ResViewFormatSignedInt4                = 18
    ResViewFormatHalf1                     = 19
    ResViewFormatHalf2                     = 20
    ResViewFormatHalf4                     = 21
    ResViewFormatFloat1                    = 22
    ResViewFormatFloat2                    = 23
    ResViewFormatFloat4                    = 24
    ResViewFormatUnsignedBlockCompressed1  = 25
    ResViewFormatUnsignedBlockCompressed2  = 26
    ResViewFormatUnsignedBlockCompressed3  = 27
    ResViewFormatUnsignedBlockCompressed4  = 28
    ResViewFormatSignedBlockCompressed4    = 29
    ResViewFormatUnsignedBlockCompressed5  = 30
    ResViewFormatSignedBlockCompressed5    = 31
    ResViewFormatUnsignedBlockCompressed6H = 32
    ResViewFormatSignedBlockCompressed6H   = 33
    ResViewFormatUnsignedBlockCompressed7  = 34
  end

  enum MemoryType
    MemoryTypeHost   = 1
    MemoryTypeDevice = 2
  end

  enum TextureAddressMode
    AddressModeWrap   = 0
    AddressModeClamp  = 1
    AddressModeMirror = 2
    AddressModeBorder = 3
  end
  enum TextureFilterMode
    FilterModePoint  = 0
    FilterModeLinear = 1
  end
  enum TextureReadMode
    ReadModeElementType     = 0
    ReadModeNormalizedFloat = 1
  end

  enum Error
    Success                          =     0
    ErrorMissingConfiguration        =     1
    ErrorMemoryAllocation            =     2
    ErrorInitializationError         =     3
    ErrorLaunchFailure               =     4
    ErrorPriorLaunchFailure          =     5
    ErrorLaunchTimeout               =     6
    ErrorLaunchOutOfResources        =     7
    ErrorInvalidDeviceFunction       =     8
    ErrorInvalidConfiguration        =     9
    ErrorInvalidDevice               =    10
    ErrorInvalidValue                =    11
    ErrorInvalidPitchValue           =    12
    ErrorInvalidSymbol               =    13
    ErrorMapBufferObjectFailed       =    14
    ErrorUnmapBufferObjectFailed     =    15
    ErrorInvalidHostPointer          =    16
    ErrorInvalidDevicePointer        =    17
    ErrorInvalidTexture              =    18
    ErrorInvalidTextureBinding       =    19
    ErrorInvalidChannelDescriptor    =    20
    ErrorInvalidMemcpyDirection      =    21
    ErrorAddressOfConstant           =    22
    ErrorTextureFetchFailed          =    23
    ErrorTextureNotBound             =    24
    ErrorSynchronizationError        =    25
    ErrorInvalidFilterSetting        =    26
    ErrorInvalidNormSetting          =    27
    ErrorMixedDeviceExecution        =    28
    ErrorCudartUnloading             =    29
    ErrorUnknown                     =    30
    ErrorNotYetImplemented           =    31
    ErrorMemoryValueTooLarge         =    32
    ErrorInvalidResourceHandle       =    33
    ErrorNotReady                    =    34
    ErrorInsufficientDriver          =    35
    ErrorSetOnActiveProcess          =    36
    ErrorInvalidSurface              =    37
    ErrorNoDevice                    =    38
    ErrorEccUncorrectable            =    39
    ErrorSharedObjectSymbolNotFound  =    40
    ErrorSharedObjectInitFailed      =    41
    ErrorUnsupportedLimit            =    42
    ErrorDuplicateVariableName       =    43
    ErrorDuplicateTextureName        =    44
    ErrorDuplicateSurfaceName        =    45
    ErrorDevicesUnavailable          =    46
    ErrorInvalidKernelImage          =    47
    ErrorNoKernelImageForDevice      =    48
    ErrorIncompatibleDriverContext   =    49
    ErrorPeerAccessAlreadyEnabled    =    50
    ErrorPeerAccessNotEnabled        =    51
    ErrorDeviceAlreadyInUse          =    54
    ErrorProfilerDisabled            =    55
    ErrorProfilerNotInitialized      =    56
    ErrorProfilerAlreadyStarted      =    57
    ErrorProfilerAlreadyStopped      =    58
    ErrorAssert                      =    59
    ErrorTooManyPeers                =    60
    ErrorHostMemoryAlreadyRegistered =    61
    ErrorHostMemoryNotRegistered     =    62
    ErrorOperatingSystem             =    63
    ErrorPeerAccessUnsupported       =    64
    ErrorLaunchMaxDepthExceeded      =    65
    ErrorLaunchFileScopedTex         =    66
    ErrorLaunchFileScopedSurf        =    67
    ErrorSyncDepthExceeded           =    68
    ErrorLaunchPendingCountExceeded  =    69
    ErrorNotPermitted                =    70
    ErrorNotSupported                =    71
    ErrorHardwareStackError          =    72
    ErrorIllegalInstruction          =    73
    ErrorMisalignedAddress           =    74
    ErrorInvalidAddressSpace         =    75
    ErrorInvalidPc                   =    76
    ErrorIllegalAddress              =    77
    ErrorInvalidPtx                  =    78
    ErrorInvalidGraphicsContext      =    79
    ErrorNvlinkUncorrectable         =    80
    ErrorJitCompilerNotFound         =    81
    ErrorCooperativeLaunchTooLarge   =    82
    ErrorStartupFailure              =   127
    ErrorApiFailureBase              = 10000
  end

  enum SharedMemConfig
    SharedMemBankSizeDefault   = 0
    SharedMemBankSizeFourByte  = 1
    SharedMemBankSizeEightByte = 2
  end
  enum Limit
    LimitStackSize                    = 0
    LimitPrintfFifoSize               = 1
    LimitMallocHeapSize               = 2
    LimitDevRuntimeSyncDepth          = 3
    LimitDevRuntimePendingLaunchCount = 4
  end
  enum FuncCache
    FuncCachePreferNone   = 0
    FuncCachePreferShared = 1
    FuncCachePreferL1     = 2
    FuncCachePreferEqual  = 3
  end

  enum DeviceP2PAttr
    DevP2PAttrPerformanceRank       = 1
    DevP2PAttrAccessSupported       = 2
    DevP2PAttrNativeAtomicSupported = 3
  end
  enum DeviceAttr
    DevAttrMaxThreadsPerBlock                =  1
    DevAttrMaxBlockDimX                      =  2
    DevAttrMaxBlockDimY                      =  3
    DevAttrMaxBlockDimZ                      =  4
    DevAttrMaxGridDimX                       =  5
    DevAttrMaxGridDimY                       =  6
    DevAttrMaxGridDimZ                       =  7
    DevAttrMaxSharedMemoryPerBlock           =  8
    DevAttrTotalConstantMemory               =  9
    DevAttrWarpSize                          = 10
    DevAttrMaxPitch                          = 11
    DevAttrMaxRegistersPerBlock              = 12
    DevAttrClockRate                         = 13
    DevAttrTextureAlignment                  = 14
    DevAttrGpuOverlap                        = 15
    DevAttrMultiProcessorCount               = 16
    DevAttrKernelExecTimeout                 = 17
    DevAttrIntegrated                        = 18
    DevAttrCanMapHostMemory                  = 19
    DevAttrComputeMode                       = 20
    DevAttrMaxTexture1DWidth                 = 21
    DevAttrMaxTexture2DWidth                 = 22
    DevAttrMaxTexture2DHeight                = 23
    DevAttrMaxTexture3DWidth                 = 24
    DevAttrMaxTexture3DHeight                = 25
    DevAttrMaxTexture3DDepth                 = 26
    DevAttrMaxTexture2DLayeredWidth          = 27
    DevAttrMaxTexture2DLayeredHeight         = 28
    DevAttrMaxTexture2DLayeredLayers         = 29
    DevAttrSurfaceAlignment                  = 30
    DevAttrConcurrentKernels                 = 31
    DevAttrEccEnabled                        = 32
    DevAttrPciBusId                          = 33
    DevAttrPciDeviceId                       = 34
    DevAttrTccDriver                         = 35
    DevAttrMemoryClockRate                   = 36
    DevAttrGlobalMemoryBusWidth              = 37
    DevAttrL2CacheSize                       = 38
    DevAttrMaxThreadsPerMultiProcessor       = 39
    DevAttrAsyncEngineCount                  = 40
    DevAttrUnifiedAddressing                 = 41
    DevAttrMaxTexture1DLayeredWidth          = 42
    DevAttrMaxTexture1DLayeredLayers         = 43
    DevAttrMaxTexture2DGatherWidth           = 45
    DevAttrMaxTexture2DGatherHeight          = 46
    DevAttrMaxTexture3DWidthAlt              = 47
    DevAttrMaxTexture3DHeightAlt             = 48
    DevAttrMaxTexture3DDepthAlt              = 49
    DevAttrPciDomainId                       = 50
    DevAttrTexturePitchAlignment             = 51
    DevAttrMaxTextureCubemapWidth            = 52
    DevAttrMaxTextureCubemapLayeredWidth     = 53
    DevAttrMaxTextureCubemapLayeredLayers    = 54
    DevAttrMaxSurface1DWidth                 = 55
    DevAttrMaxSurface2DWidth                 = 56
    DevAttrMaxSurface2DHeight                = 57
    DevAttrMaxSurface3DWidth                 = 58
    DevAttrMaxSurface3DHeight                = 59
    DevAttrMaxSurface3DDepth                 = 60
    DevAttrMaxSurface1DLayeredWidth          = 61
    DevAttrMaxSurface1DLayeredLayers         = 62
    DevAttrMaxSurface2DLayeredWidth          = 63
    DevAttrMaxSurface2DLayeredHeight         = 64
    DevAttrMaxSurface2DLayeredLayers         = 65
    DevAttrMaxSurfaceCubemapWidth            = 66
    DevAttrMaxSurfaceCubemapLayeredWidth     = 67
    DevAttrMaxSurfaceCubemapLayeredLayers    = 68
    DevAttrMaxTexture1DLinearWidth           = 69
    DevAttrMaxTexture2DLinearWidth           = 70
    DevAttrMaxTexture2DLinearHeight          = 71
    DevAttrMaxTexture2DLinearPitch           = 72
    DevAttrMaxTexture2DMipmappedWidth        = 73
    DevAttrMaxTexture2DMipmappedHeight       = 74
    DevAttrComputeCapabilityMajor            = 75
    DevAttrComputeCapabilityMinor            = 76
    DevAttrMaxTexture1DMipmappedWidth        = 77
    DevAttrStreamPrioritiesSupported         = 78
    DevAttrGlobalL1CacheSupported            = 79
    DevAttrLocalL1CacheSupported             = 80
    DevAttrMaxSharedMemoryPerMultiprocessor  = 81
    DevAttrMaxRegistersPerMultiprocessor     = 82
    DevAttrManagedMemory                     = 83
    DevAttrIsMultiGpuBoard                   = 84
    DevAttrMultiGpuBoardGroupId              = 85
    DevAttrHostNativeAtomicSupported         = 86
    DevAttrSingleToDoublePrecisionPerfRatio  = 87
    DevAttrPageableMemoryAccess              = 88
    DevAttrConcurrentManagedAccess           = 89
    DevAttrComputePreemptionSupported        = 90
    DevAttrCanUseHostPointerForRegisteredMem = 91
    DevAttrReserved92                        = 92
    DevAttrReserved93                        = 93
    DevAttrReserved94                        = 94
    DevAttrCooperativeLaunch                 = 95
    DevAttrCooperativeMultiDeviceLaunch      = 96
    DevAttrMaxSharedMemoryPerBlockOptin      = 97
  end

  enum FuncAttribute
    FuncAttributeMaxDynamicSharedMemorySize    =  8
    FuncAttributePreferredSharedMemoryCarveout =  9
    FuncAttributeMax                           = 10
  end

  struct ChannelFormatDesc
    x : LibC::Int
    y : LibC::Int
    z : LibC::Int
    w : LibC::Int
    f : ChannelFormatKind
  end

  struct PitchedPtr
    ptr : Void*
    pitch : LibC::Int
    xsize : LibC::Int
    ysize : LibC::Int
  end

  struct Extent
    width : LibC::Int
    height : LibC::Int
    depth : LibC::Int
  end

  struct Pos
    x : LibC::Int
    y : LibC::Int
    z : LibC::Int
  end

  struct Memcpy3DParms
    src_array : ArrayT
    src_pos : Pos
    src_ptr : PitchedPtr
    dst_array : ArrayT
    dst_pos : Pos
    dst_ptr : PitchedPtr
    extent : Extent
    kind : MemcpyKind
  end

  struct Memcpy3DPeerParms
    src_array : ArrayT
    src_pos : Pos
    src_ptr : PitchedPtr
    src_device : LibC::Int
    dst_array : ArrayT
    dst_pos : Pos
    dst_ptr : PitchedPtr
    dst_device : LibC::Int
    extent : Extent
  end

  struct ResourceDesc
    res_type : ResourceType
    res : ResourceDescRes
  end

  union ResourceDescRes
    array : ResourceDescResArray
    mipmap : ResourceDescResMipmap
    linear : ResourceDescResLinear
    pitch2_d : ResourceDescResPitch2D
  end

  struct ResourceDescResArray
    array : ArrayT
  end

  struct ResourceDescResMipmap
    mipmap : MipmappedArrayT
  end

  struct ResourceDescResLinear
    dev_ptr : Void*
    desc : ChannelFormatDesc
    size_in_bytes : LibC::Int
  end

  struct ResourceDescResPitch2D
    dev_ptr : Void*
    desc : ChannelFormatDesc
    width : LibC::Int
    height : LibC::Int
    pitch_in_bytes : LibC::Int
  end

  struct ResourceViewDesc
    format : ResourceViewFormat
    width : LibC::Int
    height : LibC::Int
    depth : LibC::Int
    first_mipmap_level : LibC::UInt
    last_mipmap_level : LibC::UInt
    first_layer : LibC::UInt
    last_layer : LibC::UInt
  end

  struct PointerAttributes
    memory_type : MemoryType
    device : LibC::Int
    device_pointer : Void*
    host_pointer : Void*
    is_managed : LibC::Int
  end

  struct FuncAttributes
    shared_size_bytes : LibC::Int
    const_size_bytes : LibC::Int
    local_size_bytes : LibC::Int
    max_threads_per_block : LibC::Int
    num_regs : LibC::Int
    ptx_version : LibC::Int
    binary_version : LibC::Int
    cache_mode_ca : LibC::Int
    max_dynamic_shared_size_bytes : LibC::Int
    preferred_shmem_carveout : LibC::Int
  end

  struct DeviceProp
    name : LibC::Char[256]
    total_global_mem : LibC::Int
    shared_mem_per_block : LibC::Int
    regs_per_block : LibC::Int
    warp_size : LibC::Int
    mem_pitch : LibC::Int
    max_threads_per_block : LibC::Int
    max_threads_dim : LibC::Int[3]
    max_grid_size : LibC::Int[3]
    clock_rate : LibC::Int
    total_const_mem : LibC::Int
    major : LibC::Int
    minor : LibC::Int
    texture_alignment : LibC::Int
    texture_pitch_alignment : LibC::Int
    device_overlap : LibC::Int
    multi_processor_count : LibC::Int
    kernel_exec_timeout_enabled : LibC::Int
    integrated : LibC::Int
    can_map_host_memory : LibC::Int
    compute_mode : LibC::Int
    max_texture1_d : LibC::Int
    max_texture1_d_mipmap : LibC::Int
    max_texture1_d_linear : LibC::Int
    max_texture2_d : LibC::Int[2]
    max_texture2_d_mipmap : LibC::Int[2]
    max_texture2_d_linear : LibC::Int[3]
    max_texture2_d_gather : LibC::Int[2]
    max_texture3_d : LibC::Int[3]
    max_texture3_d_alt : LibC::Int[3]
    max_texture_cubemap : LibC::Int
    max_texture1_d_layered : LibC::Int[2]
    max_texture2_d_layered : LibC::Int[3]
    max_texture_cubemap_layered : LibC::Int[2]
    max_surface1_d : LibC::Int
    max_surface2_d : LibC::Int[2]
    max_surface3_d : LibC::Int[3]
    max_surface1_d_layered : LibC::Int[2]
    max_surface2_d_layered : LibC::Int[3]
    max_surface_cubemap : LibC::Int
    max_surface_cubemap_layered : LibC::Int[2]
    surface_alignment : LibC::Int
    concurrent_kernels : LibC::Int
    ecc_enabled : LibC::Int
    pci_bus_id : LibC::Int
    pci_device_id : LibC::Int
    pci_domain_id : LibC::Int
    tcc_driver : LibC::Int
    async_engine_count : LibC::Int
    unified_addressing : LibC::Int
    memory_clock_rate : LibC::Int
    memory_bus_width : LibC::Int
    l2_cache_size : LibC::Int
    max_threads_per_multi_processor : LibC::Int
    stream_priorities_supported : LibC::Int
    global_l1_cache_supported : LibC::Int
    local_l1_cache_supported : LibC::Int
    shared_mem_per_multiprocessor : LibC::Int
    regs_per_multiprocessor : LibC::Int
    managed_memory : LibC::Int
    is_multi_gpu_board : LibC::Int
    multi_gpu_board_group_id : LibC::Int
    host_native_atomic_supported : LibC::Int
    single_to_double_precision_perf_ratio : LibC::Int
    pageable_memory_access : LibC::Int
    concurrent_managed_access : LibC::Int
    compute_preemption_supported : LibC::Int
    can_use_host_pointer_for_registered_mem : LibC::Int
    cooperative_launch : LibC::Int
    cooperative_multi_device_launch : LibC::Int
    shared_mem_per_block_optin : LibC::Int
  end

  struct IpcEventHandleSt
    reserved : LibC::Char[64]
  end

  struct IpcMemHandleSt
    reserved : LibC::Char[64]
  end

  struct LaunchParams
    func : Void*
    grid_dim : Dim3
    block_dim : Dim3
    args : Void**
    shared_mem : LibC::Int
    stream : StreamT
  end

  struct Dim3
    x : LibC::UInt
    y : LibC::UInt
    z : LibC::UInt
  end

  struct TextureDesc
    address_mode : TextureAddressMode[3]
    filter_mode : TextureFilterMode
    read_mode : TextureReadMode
    s_rgb : LibC::Int
    border_color : LibC::Float[4]
    normalized_coords : LibC::Int
    max_anisotropy : LibC::UInt
    mipmap_filter_mode : TextureFilterMode
    mipmap_level_bias : LibC::Float
    min_mipmap_level_clamp : LibC::Float
    max_mipmap_level_clamp : LibC::Float
  end

  fun device_reset = cudaDeviceReset : ErrorT
  fun device_synchronize = cudaDeviceSynchronize : ErrorT
  fun device_set_limit = cudaDeviceSetLimit(limit : Limit, value : LibC::Int) : ErrorT
  fun device_get_limit = cudaDeviceGetLimit(p_value : LibC::Int*, limit : Limit) : ErrorT
  fun device_get_cache_config = cudaDeviceGetCacheConfig(p_cache_config : FuncCache*) : ErrorT
  fun device_get_stream_priority_range = cudaDeviceGetStreamPriorityRange(least_priority : LibC::Int*, greatest_priority : LibC::Int*) : ErrorT
  fun device_set_cache_config = cudaDeviceSetCacheConfig(cache_config : FuncCache) : ErrorT
  fun device_get_shared_mem_config = cudaDeviceGetSharedMemConfig(p_config : SharedMemConfig*) : ErrorT
  fun device_set_shared_mem_config = cudaDeviceSetSharedMemConfig(config : SharedMemConfig) : ErrorT
  fun device_get_by_pci_bus_id = cudaDeviceGetByPCIBusId(device : LibC::Int*, pci_bus_id : LibC::Char*) : ErrorT
  fun device_get_pci_bus_id = cudaDeviceGetPCIBusId(pci_bus_id : LibC::Char*, len : LibC::Int, device : LibC::Int) : ErrorT
  fun ipc_get_event_handle = cudaIpcGetEventHandle(handle : IpcEventHandleT*, event : EventT) : ErrorT
  fun ipc_open_event_handle = cudaIpcOpenEventHandle(event : EventT*, handle : IpcEventHandleT) : ErrorT
  fun ipc_get_mem_handle = cudaIpcGetMemHandle(handle : IpcMemHandleT*, dev_ptr : Void*) : ErrorT
  fun ipc_open_mem_handle = cudaIpcOpenMemHandle(dev_ptr : Void**, handle : IpcMemHandleT, flags : LibC::UInt) : ErrorT
  fun ipc_close_mem_handle = cudaIpcCloseMemHandle(dev_ptr : Void*) : ErrorT
  fun thread_exit = cudaThreadExit : ErrorT
  fun thread_synchronize = cudaThreadSynchronize : ErrorT
  fun thread_set_limit = cudaThreadSetLimit(limit : Limit, value : LibC::Int) : ErrorT
  fun thread_get_limit = cudaThreadGetLimit(p_value : LibC::Int*, limit : Limit) : ErrorT
  fun thread_get_cache_config = cudaThreadGetCacheConfig(p_cache_config : FuncCache*) : ErrorT
  fun thread_set_cache_config = cudaThreadSetCacheConfig(cache_config : FuncCache) : ErrorT
  fun get_last_error = cudaGetLastError : ErrorT
  fun peek_at_last_error = cudaPeekAtLastError : ErrorT
  fun get_error_name = cudaGetErrorName(error : ErrorT) : LibC::Char*
  fun get_error_string = cudaGetErrorString(error : ErrorT) : LibC::Char*
  fun get_device_count = cudaGetDeviceCount(count : LibC::Int*) : ErrorT
  fun get_device_properties = cudaGetDeviceProperties(prop : DeviceProp*, device : LibC::Int) : ErrorT
  fun device_get_attribute = cudaDeviceGetAttribute(value : LibC::Int*, attr : DeviceAttr, device : LibC::Int) : ErrorT
  fun device_get_p2_p_attribute = cudaDeviceGetP2PAttribute(value : LibC::Int*, attr : DeviceP2PAttr, src_device : LibC::Int, dst_device : LibC::Int) : ErrorT
  fun choose_device = cudaChooseDevice(device : LibC::Int*, prop : DeviceProp*) : ErrorT
  fun set_device = cudaSetDevice(device : LibC::Int) : ErrorT
  fun get_device = cudaGetDevice(device : LibC::Int*) : ErrorT
  fun set_valid_devices = cudaSetValidDevices(device_arr : LibC::Int*, len : LibC::Int) : ErrorT
  fun set_device_flags = cudaSetDeviceFlags(flags : LibC::UInt) : ErrorT
  fun get_device_flags = cudaGetDeviceFlags(flags : LibC::UInt*) : ErrorT
  fun stream_create = cudaStreamCreate(p_stream : StreamT*) : ErrorT
  fun stream_create_with_flags = cudaStreamCreateWithFlags(p_stream : StreamT*, flags : LibC::UInt) : ErrorT
  fun stream_create_with_priority = cudaStreamCreateWithPriority(p_stream : StreamT*, flags : LibC::UInt, priority : LibC::Int) : ErrorT
  fun stream_get_priority = cudaStreamGetPriority(h_stream : StreamT, priority : LibC::Int*) : ErrorT
  fun stream_get_flags = cudaStreamGetFlags(h_stream : StreamT, flags : LibC::UInt*) : ErrorT
  fun stream_destroy = cudaStreamDestroy(stream : StreamT) : ErrorT
  fun stream_wait_event = cudaStreamWaitEvent(stream : StreamT, event : EventT, flags : LibC::UInt) : ErrorT
  fun stream_add_callback = cudaStreamAddCallback(stream : StreamT, callback : StreamCallbackT, user_data : Void*, flags : LibC::UInt) : ErrorT
  alias StreamCallbackT = (StreamT, ErrorT, Void* -> Void)
  fun stream_synchronize = cudaStreamSynchronize(stream : StreamT) : ErrorT
  fun stream_query = cudaStreamQuery(stream : StreamT) : ErrorT
  fun stream_attach_mem_async = cudaStreamAttachMemAsync(stream : StreamT, dev_ptr : Void*, length : LibC::Int, flags : LibC::UInt) : ErrorT
  fun event_create = cudaEventCreate(event : EventT*) : ErrorT
  fun event_create_with_flags = cudaEventCreateWithFlags(event : EventT*, flags : LibC::UInt) : ErrorT
  fun event_record = cudaEventRecord(event : EventT, stream : StreamT) : ErrorT
  fun event_query = cudaEventQuery(event : EventT) : ErrorT
  fun event_synchronize = cudaEventSynchronize(event : EventT) : ErrorT
  fun event_destroy = cudaEventDestroy(event : EventT) : ErrorT
  fun event_elapsed_time = cudaEventElapsedTime(ms : LibC::Float*, start : EventT, _end : EventT) : ErrorT
  fun launch_kernel = cudaLaunchKernel(func : Void*, grid_dim : Dim3, block_dim : Dim3, args : Void**, shared_mem : LibC::Int, stream : StreamT) : ErrorT
  fun launch_cooperative_kernel = cudaLaunchCooperativeKernel(func : Void*, grid_dim : Dim3, block_dim : Dim3, args : Void**, shared_mem : LibC::Int, stream : StreamT) : ErrorT
  fun launch_cooperative_kernel_multi_device = cudaLaunchCooperativeKernelMultiDevice(launch_params_list : LaunchParams*, num_devices : LibC::UInt, flags : LibC::UInt) : ErrorT
  fun func_set_cache_config = cudaFuncSetCacheConfig(func : Void*, cache_config : FuncCache) : ErrorT
  fun func_set_shared_mem_config = cudaFuncSetSharedMemConfig(func : Void*, config : SharedMemConfig) : ErrorT
  fun func_get_attributes = cudaFuncGetAttributes(attr : FuncAttributes*, func : Void*) : ErrorT
  fun func_set_attribute = cudaFuncSetAttribute(func : Void*, attr : FuncAttribute, value : LibC::Int) : ErrorT
  fun set_double_for_device = cudaSetDoubleForDevice(d : LibC::Double*) : ErrorT
  fun set_double_for_host = cudaSetDoubleForHost(d : LibC::Double*) : ErrorT
  fun occupancy_max_active_blocks_per_multiprocessor = cudaOccupancyMaxActiveBlocksPerMultiprocessor(num_blocks : LibC::Int*, func : Void*, block_size : LibC::Int, dynamic_s_mem_size : LibC::Int) : ErrorT
  fun occupancy_max_active_blocks_per_multiprocessor_with_flags = cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(num_blocks : LibC::Int*, func : Void*, block_size : LibC::Int, dynamic_s_mem_size : LibC::Int, flags : LibC::UInt) : ErrorT
  fun configure_call = cudaConfigureCall(grid_dim : Dim3, block_dim : Dim3, shared_mem : LibC::Int, stream : StreamT) : ErrorT
  fun setup_argument = cudaSetupArgument(arg : Void*, size : LibC::Int, offset : LibC::Int) : ErrorT
  fun launch = cudaLaunch(func : Void*) : ErrorT
  fun malloc_managed = cudaMallocManaged(dev_ptr : Void**, size : LibC::Int, flags : LibC::UInt) : ErrorT
  fun malloc = cudaMalloc(dev_ptr : Void**, size : LibC::Int) : ErrorT
  fun malloc_host = cudaMallocHost(ptr : Void**, size : LibC::Int) : ErrorT
  fun malloc_pitch = cudaMallocPitch(dev_ptr : Void**, pitch : LibC::Int*, width : LibC::Int, height : LibC::Int) : ErrorT
  fun malloc_array = cudaMallocArray(array : ArrayT*, desc : ChannelFormatDesc*, width : LibC::Int, height : LibC::Int, flags : LibC::UInt) : ErrorT
  fun free = cudaFree(dev_ptr : Void*) : ErrorT
  fun free_host = cudaFreeHost(ptr : Void*) : ErrorT
  fun free_array = cudaFreeArray(array : ArrayT) : ErrorT
  fun free_mipmapped_array = cudaFreeMipmappedArray(mipmapped_array : MipmappedArrayT) : ErrorT
  fun host_alloc = cudaHostAlloc(p_host : Void**, size : LibC::Int, flags : LibC::UInt) : ErrorT
  fun host_register = cudaHostRegister(ptr : Void*, size : LibC::Int, flags : LibC::UInt) : ErrorT
  fun host_unregister = cudaHostUnregister(ptr : Void*) : ErrorT
  fun host_get_device_pointer = cudaHostGetDevicePointer(p_device : Void**, p_host : Void*, flags : LibC::UInt) : ErrorT
  fun host_get_flags = cudaHostGetFlags(p_flags : LibC::UInt*, p_host : Void*) : ErrorT
  fun malloc3_d = cudaMalloc3D(pitched_dev_ptr : PitchedPtr*, extent : Extent) : ErrorT
  fun malloc3_d_array = cudaMalloc3DArray(array : ArrayT*, desc : ChannelFormatDesc*, extent : Extent, flags : LibC::UInt) : ErrorT
  fun malloc_mipmapped_array = cudaMallocMipmappedArray(mipmapped_array : MipmappedArrayT*, desc : ChannelFormatDesc*, extent : Extent, num_levels : LibC::UInt, flags : LibC::UInt) : ErrorT
  fun get_mipmapped_array_level = cudaGetMipmappedArrayLevel(level_array : ArrayT*, mipmapped_array : MipmappedArrayConstT, level : LibC::UInt) : ErrorT
  type MipmappedArrayConstT = Void*
  fun memcpy3_d = cudaMemcpy3D(p : Memcpy3DParms*) : ErrorT
  fun memcpy3_d_peer = cudaMemcpy3DPeer(p : Memcpy3DPeerParms*) : ErrorT
  fun memcpy3_d_async = cudaMemcpy3DAsync(p : Memcpy3DParms*, stream : StreamT) : ErrorT
  fun memcpy3_d_peer_async = cudaMemcpy3DPeerAsync(p : Memcpy3DPeerParms*, stream : StreamT) : ErrorT
  fun mem_get_info = cudaMemGetInfo(free : LibC::Int*, total : LibC::Int*) : ErrorT
  fun array_get_info = cudaArrayGetInfo(desc : ChannelFormatDesc*, extent : Extent*, flags : LibC::UInt*, array : ArrayT) : ErrorT
  fun memcpy = cudaMemcpy(dst : Void*, src : Void*, count : LibC::Int, kind : MemcpyKind) : ErrorT
  fun memcpy_peer = cudaMemcpyPeer(dst : Void*, dst_device : LibC::Int, src : Void*, src_device : LibC::Int, count : LibC::Int) : ErrorT
  fun memcpy_to_array = cudaMemcpyToArray(dst : ArrayT, w_offset : LibC::Int, h_offset : LibC::Int, src : Void*, count : LibC::Int, kind : MemcpyKind) : ErrorT
  fun memcpy_from_array = cudaMemcpyFromArray(dst : Void*, src : ArrayConstT, w_offset : LibC::Int, h_offset : LibC::Int, count : LibC::Int, kind : MemcpyKind) : ErrorT
  type ArrayConstT = Void*
  fun memcpy_array_to_array = cudaMemcpyArrayToArray(dst : ArrayT, w_offset_dst : LibC::Int, h_offset_dst : LibC::Int, src : ArrayConstT, w_offset_src : LibC::Int, h_offset_src : LibC::Int, count : LibC::Int, kind : MemcpyKind) : ErrorT
  fun memcpy2_d = cudaMemcpy2D(dst : Void*, dpitch : LibC::Int, src : Void*, spitch : LibC::Int, width : LibC::Int, height : LibC::Int, kind : MemcpyKind) : ErrorT
  fun memcpy2_d_to_array = cudaMemcpy2DToArray(dst : ArrayT, w_offset : LibC::Int, h_offset : LibC::Int, src : Void*, spitch : LibC::Int, width : LibC::Int, height : LibC::Int, kind : MemcpyKind) : ErrorT
  fun memcpy2_d_from_array = cudaMemcpy2DFromArray(dst : Void*, dpitch : LibC::Int, src : ArrayConstT, w_offset : LibC::Int, h_offset : LibC::Int, width : LibC::Int, height : LibC::Int, kind : MemcpyKind) : ErrorT
  fun memcpy2_d_array_to_array = cudaMemcpy2DArrayToArray(dst : ArrayT, w_offset_dst : LibC::Int, h_offset_dst : LibC::Int, src : ArrayConstT, w_offset_src : LibC::Int, h_offset_src : LibC::Int, width : LibC::Int, height : LibC::Int, kind : MemcpyKind) : ErrorT
  fun memcpy_to_symbol = cudaMemcpyToSymbol(symbol : Void*, src : Void*, count : LibC::Int, offset : LibC::Int, kind : MemcpyKind) : ErrorT
  fun memcpy_from_symbol = cudaMemcpyFromSymbol(dst : Void*, symbol : Void*, count : LibC::Int, offset : LibC::Int, kind : MemcpyKind) : ErrorT
  fun memcpy_async = cudaMemcpyAsync(dst : Void*, src : Void*, count : LibC::Int, kind : MemcpyKind, stream : StreamT) : ErrorT
  fun memcpy_peer_async = cudaMemcpyPeerAsync(dst : Void*, dst_device : LibC::Int, src : Void*, src_device : LibC::Int, count : LibC::Int, stream : StreamT) : ErrorT
  fun memcpy_to_array_async = cudaMemcpyToArrayAsync(dst : ArrayT, w_offset : LibC::Int, h_offset : LibC::Int, src : Void*, count : LibC::Int, kind : MemcpyKind, stream : StreamT) : ErrorT
  fun memcpy_from_array_async = cudaMemcpyFromArrayAsync(dst : Void*, src : ArrayConstT, w_offset : LibC::Int, h_offset : LibC::Int, count : LibC::Int, kind : MemcpyKind, stream : StreamT) : ErrorT
  fun memcpy2_d_async = cudaMemcpy2DAsync(dst : Void*, dpitch : LibC::Int, src : Void*, spitch : LibC::Int, width : LibC::Int, height : LibC::Int, kind : MemcpyKind, stream : StreamT) : ErrorT
  fun memcpy2_d_to_array_async = cudaMemcpy2DToArrayAsync(dst : ArrayT, w_offset : LibC::Int, h_offset : LibC::Int, src : Void*, spitch : LibC::Int, width : LibC::Int, height : LibC::Int, kind : MemcpyKind, stream : StreamT) : ErrorT
  fun memcpy2_d_from_array_async = cudaMemcpy2DFromArrayAsync(dst : Void*, dpitch : LibC::Int, src : ArrayConstT, w_offset : LibC::Int, h_offset : LibC::Int, width : LibC::Int, height : LibC::Int, kind : MemcpyKind, stream : StreamT) : ErrorT
  fun memcpy_to_symbol_async = cudaMemcpyToSymbolAsync(symbol : Void*, src : Void*, count : LibC::Int, offset : LibC::Int, kind : MemcpyKind, stream : StreamT) : ErrorT
  fun memcpy_from_symbol_async = cudaMemcpyFromSymbolAsync(dst : Void*, symbol : Void*, count : LibC::Int, offset : LibC::Int, kind : MemcpyKind, stream : StreamT) : ErrorT
  fun memset = cudaMemset(dev_ptr : Void*, value : LibC::Int, count : LibC::Int) : ErrorT
  fun memset2_d = cudaMemset2D(dev_ptr : Void*, pitch : LibC::Int, value : LibC::Int, width : LibC::Int, height : LibC::Int) : ErrorT
  fun memset3_d = cudaMemset3D(pitched_dev_ptr : PitchedPtr, value : LibC::Int, extent : Extent) : ErrorT
  fun memset_async = cudaMemsetAsync(dev_ptr : Void*, value : LibC::Int, count : LibC::Int, stream : StreamT) : ErrorT
  fun memset2_d_async = cudaMemset2DAsync(dev_ptr : Void*, pitch : LibC::Int, value : LibC::Int, width : LibC::Int, height : LibC::Int, stream : StreamT) : ErrorT
  fun memset3_d_async = cudaMemset3DAsync(pitched_dev_ptr : PitchedPtr, value : LibC::Int, extent : Extent, stream : StreamT) : ErrorT
  fun get_symbol_address = cudaGetSymbolAddress(dev_ptr : Void**, symbol : Void*) : ErrorT
  fun get_symbol_size = cudaGetSymbolSize(size : LibC::Int*, symbol : Void*) : ErrorT
  fun mem_prefetch_async = cudaMemPrefetchAsync(dev_ptr : Void*, count : LibC::Int, dst_device : LibC::Int, stream : StreamT) : ErrorT
  fun mem_advise = cudaMemAdvise(dev_ptr : Void*, count : LibC::Int, advice : MemoryAdvise, device : LibC::Int) : ErrorT
  enum MemoryAdvise
    MemAdviseSetReadMostly          = 1
    MemAdviseUnsetReadMostly        = 2
    MemAdviseSetPreferredLocation   = 3
    MemAdviseUnsetPreferredLocation = 4
    MemAdviseSetAccessedBy          = 5
    MemAdviseUnsetAccessedBy        = 6
  end
  fun mem_range_get_attribute = cudaMemRangeGetAttribute(data : Void*, data_size : LibC::Int, attribute : MemRangeAttribute, dev_ptr : Void*, count : LibC::Int) : ErrorT
  enum MemRangeAttribute
    MemRangeAttributeReadMostly           = 1
    MemRangeAttributePreferredLocation    = 2
    MemRangeAttributeAccessedBy           = 3
    MemRangeAttributeLastPrefetchLocation = 4
  end
  fun mem_range_get_attributes = cudaMemRangeGetAttributes(data : Void**, data_sizes : LibC::Int*, attributes : MemRangeAttribute*, num_attributes : LibC::Int, dev_ptr : Void*, count : LibC::Int) : ErrorT
  fun pointer_get_attributes = cudaPointerGetAttributes(attributes : PointerAttributes*, ptr : Void*) : ErrorT
  fun device_can_access_peer = cudaDeviceCanAccessPeer(can_access_peer : LibC::Int*, device : LibC::Int, peer_device : LibC::Int) : ErrorT
  fun device_enable_peer_access = cudaDeviceEnablePeerAccess(peer_device : LibC::Int, flags : LibC::UInt) : ErrorT
  fun device_disable_peer_access = cudaDeviceDisablePeerAccess(peer_device : LibC::Int) : ErrorT
  fun graphics_unregister_resource = cudaGraphicsUnregisterResource(resource : GraphicsResourceT) : ErrorT
  type GraphicsResourceT = Void*
  fun graphics_resource_set_map_flags = cudaGraphicsResourceSetMapFlags(resource : GraphicsResourceT, flags : LibC::UInt) : ErrorT
  fun graphics_map_resources = cudaGraphicsMapResources(count : LibC::Int, resources : GraphicsResourceT*, stream : StreamT) : ErrorT
  fun graphics_unmap_resources = cudaGraphicsUnmapResources(count : LibC::Int, resources : GraphicsResourceT*, stream : StreamT) : ErrorT
  fun graphics_resource_get_mapped_pointer = cudaGraphicsResourceGetMappedPointer(dev_ptr : Void**, size : LibC::Int*, resource : GraphicsResourceT) : ErrorT
  fun graphics_sub_resource_get_mapped_array = cudaGraphicsSubResourceGetMappedArray(array : ArrayT*, resource : GraphicsResourceT, array_index : LibC::UInt, mip_level : LibC::UInt) : ErrorT
  fun graphics_resource_get_mapped_mipmapped_array = cudaGraphicsResourceGetMappedMipmappedArray(mipmapped_array : MipmappedArrayT*, resource : GraphicsResourceT) : ErrorT
  fun get_channel_desc = cudaGetChannelDesc(desc : ChannelFormatDesc*, array : ArrayConstT) : ErrorT
  fun create_channel_desc = cudaCreateChannelDesc(x : LibC::Int, y : LibC::Int, z : LibC::Int, w : LibC::Int, f : ChannelFormatKind) : ChannelFormatDesc
  fun bind_texture = cudaBindTexture(offset : LibC::Int*, texref : TextureReference*, dev_ptr : Void*, desc : ChannelFormatDesc*, size : LibC::Int) : ErrorT

  struct TextureReference
    normalized : LibC::Int
    filter_mode : TextureFilterMode
    address_mode : TextureAddressMode[3]
    channel_desc : ChannelFormatDesc
    s_rgb : LibC::Int
    max_anisotropy : LibC::UInt
    mipmap_filter_mode : TextureFilterMode
    mipmap_level_bias : LibC::Float
    min_mipmap_level_clamp : LibC::Float
    max_mipmap_level_clamp : LibC::Float
    __cuda_reserved : LibC::Int[15]
  end

  fun bind_texture2_d = cudaBindTexture2D(offset : LibC::Int*, texref : TextureReference*, dev_ptr : Void*, desc : ChannelFormatDesc*, width : LibC::Int, height : LibC::Int, pitch : LibC::Int) : ErrorT
  fun bind_texture_to_array = cudaBindTextureToArray(texref : TextureReference*, array : ArrayConstT, desc : ChannelFormatDesc*) : ErrorT
  fun bind_texture_to_mipmapped_array = cudaBindTextureToMipmappedArray(texref : TextureReference*, mipmapped_array : MipmappedArrayConstT, desc : ChannelFormatDesc*) : ErrorT
  fun unbind_texture = cudaUnbindTexture(texref : TextureReference*) : ErrorT
  fun get_texture_alignment_offset = cudaGetTextureAlignmentOffset(offset : LibC::Int*, texref : TextureReference*) : ErrorT
  fun get_texture_reference = cudaGetTextureReference(texref : TextureReference**, symbol : Void*) : ErrorT
  fun bind_surface_to_array = cudaBindSurfaceToArray(surfref : SurfaceReference*, array : ArrayConstT, desc : ChannelFormatDesc*) : ErrorT

  struct SurfaceReference
    channel_desc : ChannelFormatDesc
  end

  fun get_surface_reference = cudaGetSurfaceReference(surfref : SurfaceReference**, symbol : Void*) : ErrorT
  fun create_texture_object = cudaCreateTextureObject(p_tex_object : TextureObjectT*, p_res_desc : ResourceDesc*, p_tex_desc : TextureDesc*, p_res_view_desc : ResourceViewDesc*) : ErrorT
  alias TextureObjectT = LibC::ULongLong
  fun destroy_texture_object = cudaDestroyTextureObject(tex_object : TextureObjectT) : ErrorT
  fun get_texture_object_resource_desc = cudaGetTextureObjectResourceDesc(p_res_desc : ResourceDesc*, tex_object : TextureObjectT) : ErrorT
  fun get_texture_object_texture_desc = cudaGetTextureObjectTextureDesc(p_tex_desc : TextureDesc*, tex_object : TextureObjectT) : ErrorT
  fun get_texture_object_resource_view_desc = cudaGetTextureObjectResourceViewDesc(p_res_view_desc : ResourceViewDesc*, tex_object : TextureObjectT) : ErrorT
  fun create_surface_object = cudaCreateSurfaceObject(p_surf_object : SurfaceObjectT*, p_res_desc : ResourceDesc*) : ErrorT
  alias SurfaceObjectT = LibC::ULongLong
  fun destroy_surface_object = cudaDestroySurfaceObject(surf_object : SurfaceObjectT) : ErrorT
  fun get_surface_object_resource_desc = cudaGetSurfaceObjectResourceDesc(p_res_desc : ResourceDesc*, surf_object : SurfaceObjectT) : ErrorT
  fun driver_get_version = cudaDriverGetVersion(driver_version : LibC::Int*) : ErrorT
  fun runtime_get_version = cudaRuntimeGetVersion(runtime_version : LibC::Int*) : ErrorT
  fun get_export_table = cudaGetExportTable(pp_export_table : Void**, p_export_table_id : UuidT) : ErrorT
  type UuidT = Void*
end
