require "./jet/**"

module Jet
  extend self

  def check_success(status : LibCUDA::ErrorT)
    raise status.to_s unless LibCUDA::ErrorT::Success == status
  end

  def malloc(size) : Pointer(Void)
    check_success(LibCUDA.malloc(out ptr, size))
    ptr
  end

  def free(ptr)
    check_success(LibCUDA.free(ptr))
  end
end
