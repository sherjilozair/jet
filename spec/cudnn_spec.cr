require "./spec_helper"

describe CuDNN do
  # TODO: Write tests

  it "creates a handler" do
    handler = CuDNN::Handler.new
    handler.handler.should be_a(LibCuDNN::HandleT)
    handler.destroy
  end

  it "creates a tensor descriptor" do
    t = CuDNN::Tensor4D.new(LibCuDNN::TensorFormatT::TensorNhwc,
      LibCuDNN::DataTypeT::DataFloat, 8, 64, 3, 3)
    t.desc.should be_a(LibCuDNN::TensorDescriptorT)
    t.destroy
  end
end
