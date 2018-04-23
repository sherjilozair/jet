module CuDNN
  class Handler
    getter :handler

    def initialize
      check_success(LibCuDNN.create(out @handler))
    end

    def check_success(status)
      raise status.to_s unless LibCuDNN::StatusT::Success == status
    end

    def destroy
      LibCuDNN.destroy(@handler)
    end
  end

  class Tensor4D
    getter :desc

    def initialize(
      @format : LibCuDNN::TensorFormatT,
      @data_type : LibCuDNN::DataTypeT,
      @n : LibC::Int,
      @c : LibC::Int,
      @h : LibC::Int,
      @w : LibC::Int
    )
      check_success(LibCuDNN.create_tensor_descriptor(out @desc))
      check_success(LibCuDNN.set_tensor4d_descriptor(@desc, @format, @data_type, @n, @c, @h, @w))
    end

    def check_success(status)
      raise status.to_s unless LibCuDNN::StatusT::Success == status
    end

    def destroy
      LibCuDNN.destroy_tensor_descriptor(@desc)
    end
  end
end
