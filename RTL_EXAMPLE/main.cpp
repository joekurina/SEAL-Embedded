#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include "the_fft_sycl.hpp"
#include "fft_io.hpp"

using namespace sycl;

// Use the types from the header file directly
// using FFT_Input = the_fft_input_t;
// using FFT_Output = the_fft_output_t;

// Ensure the sizes are correct
static_assert(sizeof(FFT_Input_Data) == 64);
static_assert(sizeof(FFT_Output_Data) == 64);

class FFTTest;

constexpr size_t kSize = 1024; // 4 points per struct = 4096 total points

#if FPGA_SIMULATOR
auto selector = sycl::ext::intel::fpga_simulator_selector_v;
#elif FPGA_HARDWARE
auto selector = sycl::ext::intel::fpga_selector_v;
#else  // #if FPGA_EMULATOR
auto selector = sycl::ext::intel::fpga_emulator_selector_v;
#endif

// FFT kernels
class InputKernel;
class OutputKernel;
class FFTKernel;
class FFTInputPipeName;
class FFTOutputPipeName;

// Pipe Definitions
// Pipes hold 1024 FFT_Input/Output_Data structures, 4096 total points
using FFTInputPipe = sycl::ext::intel::pipe<FFTInputPipeName, FFT_Input_Data, kSize>;
using FFTOutputPipe = sycl::ext::intel::pipe<FFTOutputPipeName, FFT_Output_Data, kSize>;

class InputKernel {
    private:
        buffer<FFT_Input_Data, 1> &input_buffer_;

    public:
        // Constructor accepting the input data buffer
        InputKernel( buffer<FFT_Input_Data, 1> &input_buffer )
            : input_buffer_(input_buffer) {}
        // Kernel function to read from the input buffer and write to the pipe
        void operator()(sycl::handler& h) const {
            // Access the input buffer
            auto data = input_buffer_.get_access<sycl::access::mode::read>(h);

            // write data to the pipe
            h.single_task([=]() [[intel::kernel_args_restrict]] {
                for (size_t i = 0; i < kSize; i++) {
                    FFTInputPipe::write(data[i]);
                }
            });
        }
};

class OutputKernel {
    private:
        buffer<FFT_Output_Data, 1> &output_buffer_;

    public:
        // Constructor accepting the output data buffer
        OutputKernel( buffer<FFT_Output_Data, 1> &output_buffer )
            : output_buffer_(output_buffer) {}
        // Kernel function to read from the pipe and write to the output buffer
        void operator()(sycl::handler& h) const {
            // Access the output buffer
            auto data = output_buffer_.get_access<sycl::access::mode::write>(h);

            // read data from the pipe and write to the output buffer
            h.single_task([=]() [[intel::kernel_args_restrict]] {
                for (size_t i = 0; i < kSize; i++) {
                    data[i] = FFTOutputPipe::read();
                }
            });
        }
};

class FFTKernel {
    private:
    
    public:
        void operator()(sycl::handler& h) const {
            h.single_task<FFTKernel>([=]() [[intel::kernel_args_restrict]] {
#ifdef FPGA_EMULATOR
                fft_example_DUT* instance = the_fft_new_instance();
#endif
                // size_t outputs_written = 0;
                
                [[intel::initiation_interval(1)]]
                while (1) { // Could also be while (outputs_written < kSize)
                    bool valid = false;
                    FFT_Input_Data input_data = FFTInputPipe::read(valid);
                    
                    the_fft_input_t input;
                    input.port_v_in_s = valid;  // Set valid flag based on pipe read
                    input.port_channel_in_s = 1;  // Channel 1
                    input.port_data_in_0re = input_data.port_data_in_0re;
                    input.port_data_in_0im = input_data.port_data_in_0im;
                    input.port_data_in_1re = input_data.port_data_in_1re;
                    input.port_data_in_1im = input_data.port_data_in_1im;
                    input.port_data_in_2re = input_data.port_data_in_2re;
                    input.port_data_in_2im = input_data.port_data_in_2im;
                    input.port_data_in_3re = input_data.port_data_in_3re;
                    input.port_data_in_3im = input_data.port_data_in_3im;
                            
#ifdef FPGA_EMULATOR
                    the_fft_output_t output = the_fft(instance, input);
#else
                    the_fft_output_t output = the_fft(input);
#endif
                    
                    // Check if the output is valid and write it to output pipe
                    if (output.port_v_out_s == 1) {
                        FFT_Output_Data data;
                        data.port_data_out_0re = output.port_data_out_0re;
                        data.port_data_out_0im = output.port_data_out_0im;
                        data.port_data_out_1re = output.port_data_out_1re;
                        data.port_data_out_1im = output.port_data_out_1im;
                        data.port_data_out_2re = output.port_data_out_2re;
                        data.port_data_out_2im = output.port_data_out_2im;
                        data.port_data_out_3re = output.port_data_out_3re;
                        data.port_data_out_3im = output.port_data_out_3im;
                        FFTOutputPipe::write(data);
                        //outputs_written++;
                    }
                }
                
#ifdef FPGA_EMULATOR
                the_fft_delete_instance(instance);
#endif
            });
        }
};

int main() {
    // Create buffers
    buffer<FFT_Input_Data, 1> input_buffer(fft_test_input, range<1>(kSize));
    buffer<FFT_Output_Data, 1> output_buffer{range<1>(kSize)};
    
    // Print first 5 input structs
    std::cout << "First 5 input structs:" << std::endl;
    for (int i = 0; i < 5; i++) {
        std::cout << "Input[" << i << "]: ("
            << fft_test_input[i].port_data_in_0re << ", "
            << fft_test_input[i].port_data_in_0im << ", "
            << fft_test_input[i].port_data_in_1re << ", "
            << fft_test_input[i].port_data_in_1im << ", "
            << fft_test_input[i].port_data_in_2re << ", "
            << fft_test_input[i].port_data_in_2im << ", "
            << fft_test_input[i].port_data_in_3re << ", "
            << fft_test_input[i].port_data_in_3im
            << ")" << std::endl;
    }
    std::cout << std::endl;
    
    try {
        sycl::property_list queue_properties{sycl::property::queue::enable_profiling()};
        queue q(selector, queue_properties);
        
        std::cout << "Running on device: " << q.get_device().get_info<sycl::info::device::name>() << std::endl;
        
        // Submit kernels
        auto output_event = q.submit([&](handler& h) { 
            OutputKernel kernel(output_buffer); 
            kernel(h); 
        });
        
        q.submit([&](handler& h) { 
            InputKernel kernel(input_buffer); 
            kernel(h); 
        });
        
        q.submit([&](handler& h) { 
            FFTKernel kernel; 
            kernel(h); 
        });
        
        
        // Wait for completion
        output_event.wait();
        
        std::cout << "FFT processing completed successfully!" << std::endl;
        
        // Print first 20 output structs to see valid data
        auto output_accessor = output_buffer.get_host_access();
        std::cout << "\nFirst 20 output structs:" << std::endl;
        for (int i = 0; i < 20; i++) {
            std::cout << "Output[" << i << "]: ("
                << output_accessor[i].port_data_out_0re << ", "
                << output_accessor[i].port_data_out_0im << ", "
                << output_accessor[i].port_data_out_1re << ", "
                << output_accessor[i].port_data_out_1im << ", "
                << output_accessor[i].port_data_out_2re << ", "
                << output_accessor[i].port_data_out_2im << ", "
                << output_accessor[i].port_data_out_3re << ", "
                << output_accessor[i].port_data_out_3im
                << ")" << std::endl;
        }
        
    } catch (sycl::exception const& e) {
        std::cerr << "SYCL exception caught: " << e.what() << std::endl;
        return 1;
    } catch (std::exception const& e) {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}