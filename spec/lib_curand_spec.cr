require "./spec_helper"

describe "LibCURAND" do
  it "create generator" do
    error = LibCURAND.create_generator(out generator, LibCURAND::RngType::RngPseudoDefault)
    error.should be_a(LibCURAND::Status::Success)
    LibCURAND.destroy_generator(generator)
  end

  it "generate random number" do
    LibCURAND.create_generator(out generator, LibCURAND::RngType::RngPseudoDefault)
    error = LibCURAND.generate(generator, out rand_num1, 5)
    error = LibCURAND.generate(generator, out rand_num2, 5)
    rand_num1.should_not eq(rand_num2)
    LibCURAND.destroy_generator(generator)
  end
end
