module CuDNN
  class Handler
    getter :handler

    def initialize
      check_success(LibCuDNN.create(out @handler))
    end

    def check_success(status)
      raise status.to_s unless LibCuDNN::StatusT.new(0) == status
    end

    def destroy
      LibCuDNN.destroy(@handler)
    end
  end

  class TensorDescriptor
    getter :td

    def initialize
      check_success(LibCuDNN.create_tensor_descriptor(out @td))
    end

    def check_success(status)
      raise status.to_s unless LibCuDNN::StatusT.new(0) == status
    end

    def destroy
      LibCuDNN.destroy_tensor_descriptor(@td)
    end
  end
end
