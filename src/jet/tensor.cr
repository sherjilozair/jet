module Jet
  class Tensor
    property :desc

    @strides : Array(Int32)

    private def get_strides(dims)
      strides = [] of Int32
      last = 1
      dims.reverse.each do |dim|
        strides << last
        last *= dim
      end
      strides.reverse!
    end

    def initialize(@data_type : LibCuDNN::DataTypeT, @dims : Array(Int32))
      @strides = get_strides(@dims)
      check_success(LibCuDNN.create_tensor_descriptor(out @desc))
      check_success(LibCuDNN.set_tensor_nd_descriptor(@desc, @data_type, @dims.size, @dims.to_unsafe, @strides.to_unsafe))
      check_success(LibCuDNN.get_tensor_size_in_bytes(@desc, out size))
      check_success(LibCUDA.malloc(out @ptr, size))
      check_success(LibCuDNN.set_tensor(@handle, @desc, @ptr)) # TODO: need to figure out where handle will live
    end

    def check_success(status : LibCuDNN::StatusT)
      raise status.to_s unless LibCuDNN::StatusT::Success == status
    end

    def check_success(status : LibCUDA::ErrorT)
      raise status.to_s unless LibCUDA::ErrorT::Success == status
    end

    def destroy
      LibCuDNN.destroy_tensor_descriptor(@desc)
      check_success(LibCUDA.free(@ptr))
    end
  end
end
