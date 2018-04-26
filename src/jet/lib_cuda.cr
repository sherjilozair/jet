@[Link("cudart")]
lib LibCUDA
  enum ErrorT
    Success
  end
  fun malloc = cudaMalloc(ptr : Void**, size : LibC::Int) : ErrorT
  fun free = cudaFree(ptr : Void*) : ErrorT
  fun device_sync = cudaDeviceSynchronize : ErrorT
  fun check_error = cudaGetLastError : ErrorT
end
