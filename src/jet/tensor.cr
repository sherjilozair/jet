module Jet
  extend self
  class Tensor
    def initialize(@data_type : LibCuDNN::DataTypeT, @dims : Array(Int))
      check_success(LibCuDNN.create_tensor_descriptor(out @desc)) 
      check_success(LibCuDNN.set_tensor_nd_descriptor_ex(@desc, @data_type, @dims.size, @dims.to_unsafe))
    end
    
    def check_success(status : LibCuDNN::StatusT)
      raise status.to_s unless LibCuDNN::StatusT::Success == status
    end

    def finalize
    end
  end
end

