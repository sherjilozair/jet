module CuDNN
  class Handler
    getter :handler
    
    def initialize
      check_error(LibCuDNN.create(out @handler))
    end

    def check_error(status)
      raise status.to_s unless LibCuDNN::StatusT.new(0) == status
    end

    def destroy
      LibCuDNN.destroy(@handler)
    end
  end
end