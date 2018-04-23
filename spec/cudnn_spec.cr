require "./spec_helper"

describe CuDNN do
  # TODO: Write tests

  it "creates a handler" do
    handler = CuDNN::Handler.new
    handler.handler.should be_a(LibCuDNN::HandleT)
    handler.destroy
  end

  it "creates a tensor descriptor" do
    td = CuDNN::TensorDescriptor.new
    td.td.should be_a(LibCuDNN::TensorDescriptorT)
    td.destroy
  end
end
