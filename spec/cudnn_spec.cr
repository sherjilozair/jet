require "./spec_helper"

describe CuDNN do
  it "creates a handler" do
    handler = CuDNN::Handler.new
    handler.handler.should be_a(LibCuDNN::HandleT)
    handler.destroy
  end

  it "creates a tensor descriptor" do
    t = CuDNN::Tensor4D.new(LibCuDNN::TensorFormatT::Nhwc,
      LibCuDNN::DataTypeT::Float, 8, 64, 3, 3)
    t.desc.should be_a(LibCuDNN::TensorDescriptorT)
    t.destroy
  end

  it "allocates memory" do
    p = ::CuDNN.malloc(100)
    p.should be_a(Pointer(Void))
    ::CuDNN.free(p)
  end
end
