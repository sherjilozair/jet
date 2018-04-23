@[Link("cudart")]
lib LibCUDA
  enum ErrorT
    Success
  end
  fun malloc = cudaMalloc(ptr : Void**, size : LibC::Int) : ErrorT
  fun free = cudaFree(ptr : Void*) : ErrorT
end
