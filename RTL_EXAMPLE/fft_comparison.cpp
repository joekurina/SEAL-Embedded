#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <iostream>
#include <iomanip>
#include <complex>
#include <vector>
#include <cmath>
#include <fstream>

#include "SYCL_fft.h"
#include "the_fft_sycl.hpp"
#include "fft_io.hpp"

using namespace sycl;

#if FPGA_SIMULATOR
auto selector = sycl::ext::intel::fpga_simulator_selector_v;
#elif FPGA_HARDWARE
auto selector = sycl::ext::intel::fpga_selector_v;
#else  // FPGA_EMULATOR
auto selector = sycl::ext::intel::fpga_emulator_selector_v;
#endif

// Kernel class declarations (from main.cpp)
constexpr size_t kSize = 1024;
class InputKernel;
class OutputKernel;
class FFTKernel;
class FFTInputPipeName;
class FFTOutputPipeName;

// Pipe Definitions (from main.cpp)
using FFTInputPipe = sycl::ext::intel::pipe<FFTInputPipeName, FFT_Input_Data, kSize>;
using FFTOutputPipe = sycl::ext::intel::pipe<FFTOutputPipeName, FFT_Output_Data, kSize>;

// Kernel implementations (from main.cpp)
class InputKernel {
    private:
        buffer<FFT_Input_Data, 1> &input_buffer_;

    public:
        InputKernel( buffer<FFT_Input_Data, 1> &input_buffer )
            : input_buffer_(input_buffer) {}
        void operator()(sycl::handler& h) const {
            auto data = input_buffer_.get_access<sycl::access::mode::read>(h);
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
        OutputKernel( buffer<FFT_Output_Data, 1> &output_buffer )
            : output_buffer_(output_buffer) {}
        void operator()(sycl::handler& h) const {
            auto data = output_buffer_.get_access<sycl::access::mode::write>(h);
            h.single_task([=]() [[intel::kernel_args_restrict]] {
                for (size_t i = 0; i < kSize; i++) {
                    data[i] = FFTOutputPipe::read();
                }
            });
        }
};

class FFTKernel {
    public:
        void operator()(sycl::handler& h) const {
            h.single_task<FFTKernel>([=]() [[intel::kernel_args_restrict]] {
#ifdef FPGA_EMULATOR
                fft_example_DUT* instance = the_fft_new_instance();
#endif
                
                [[intel::initiation_interval(1)]]
                while (1) {
                    bool valid = false;
                    FFT_Input_Data input_data = FFTInputPipe::read(valid);
                    
                    the_fft_input_t input;
                    input.port_v_in_s = valid;
                    input.port_channel_in_s = 1;
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
                    }
                }
                
#ifdef FPGA_EMULATOR
                the_fft_delete_instance(instance);
#endif
            });
        }
};

// Convert FFT_Input_Data array to complex<double> array for SYCL FFT
std::vector<std::complex<double>> convert_to_complex_array(const FFT_Input_Data* input_data, size_t num_structs) {
    std::vector<std::complex<double>> result;
    result.reserve(num_structs * 4); // 4 complex numbers per struct
    
    for (size_t i = 0; i < num_structs; i++) {
        result.emplace_back(input_data[i].port_data_in_0re, input_data[i].port_data_in_0im);
        result.emplace_back(input_data[i].port_data_in_1re, input_data[i].port_data_in_1im);
        result.emplace_back(input_data[i].port_data_in_2re, input_data[i].port_data_in_2im);
        result.emplace_back(input_data[i].port_data_in_3re, input_data[i].port_data_in_3im);
    }
    
    return result;
}

// Convert RTL output to complex<double> array for comparison
std::vector<std::complex<double>> convert_rtl_output_to_complex(const FFT_Output_Data* output_data, size_t num_structs) {
    std::vector<std::complex<double>> result;
    result.reserve(num_structs * 4);
    
    for (size_t i = 0; i < num_structs; i++) {
        result.emplace_back(output_data[i].port_data_out_0re, output_data[i].port_data_out_0im);
        result.emplace_back(output_data[i].port_data_out_1re, output_data[i].port_data_out_1im);
        result.emplace_back(output_data[i].port_data_out_2re, output_data[i].port_data_out_2im);
        result.emplace_back(output_data[i].port_data_out_3re, output_data[i].port_data_out_3im);
    }
    
    return result;
}

// Run RTL FFT using exact same approach as main.cpp
std::vector<FFT_Output_Data> run_rtl_fft(const FFT_Input_Data* input_data, size_t num_structs) {
    std::vector<FFT_Output_Data> rtl_outputs(num_structs);
    
    try {
        sycl::property_list queue_properties{sycl::property::queue::enable_profiling()};
        queue q(selector, queue_properties);
        
        // Create buffers exactly like main.cpp
        buffer<FFT_Input_Data, 1> input_buffer(input_data, range<1>(kSize));
        buffer<FFT_Output_Data, 1> output_buffer{range<1>(kSize)};
        
        // Submit kernels exactly like main.cpp
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
        
        // Copy results
        auto output_accessor = output_buffer.get_host_access();
        for (size_t i = 0; i < num_structs; i++) {
            rtl_outputs[i] = output_accessor[i];
        }
        
    } catch (sycl::exception const& e) {
        std::cerr << "SYCL exception in RTL FFT: " << e.what() << std::endl;
    }
    
    return rtl_outputs;
}

// Run SYCL IFFT
std::vector<std::complex<double>> run_sycl_ifft(const std::vector<std::complex<double>>& input_data) {
    const size_t n = input_data.size();
    const size_t logn = static_cast<size_t>(std::log2(n));
    
    // Create a copy of input data for processing
    std::vector<std::complex<double>> data = input_data;
    
    try {
        queue q(selector);
        
        // Create buffer for the data
        buffer<std::complex<double>, 1> data_buffer(data.data(), range<1>(n));
        
        // Create and run the IFFT kernel
        IFFTKernel kernel(n, logn, data_buffer);
        q.submit([&](handler& h) {
            kernel(h);
        });
        
        // Wait for completion and get results
        q.wait();
        
        // Copy results back
        auto host_accessor = data_buffer.get_host_access();
        for (size_t i = 0; i < n; i++) {
            data[i] = host_accessor[i];
        }
        
    } catch (sycl::exception const& e) {
        std::cerr << "SYCL exception in IFFT: " << e.what() << std::endl;
    }
    
    return data;
}

// Compare two complex arrays and print differences
void compare_outputs(const std::vector<std::complex<double>>& sycl_output, 
                    const std::vector<std::complex<double>>& rtl_output,
                    double tolerance = 1e-10) {
    
    if (sycl_output.size() != rtl_output.size()) {
        std::cout << "Error: Output sizes don't match! SYCL: " << sycl_output.size() 
                  << ", RTL: " << rtl_output.size() << std::endl;
        return;
    }
    
    std::cout << std::fixed << std::setprecision(12);
    std::cout << "\n=== FFT Output Comparison ===\n";
    std::cout << "Index\t\tSYCL IFFT Output\t\t\tRTL FFT Output\t\t\t\tDifference\n";
    std::cout << "-----\t\t----------------\t\t\t--------------\t\t\t\t----------\n";
    
    double max_diff = 0.0;
    size_t max_diff_idx = 0;
    size_t differences_count = 0;
    
    for (size_t i = 0; i < std::min(sycl_output.size(), static_cast<size_t>(20)); i++) {
        std::complex<double> diff = sycl_output[i] - rtl_output[i];
        double magnitude_diff = std::abs(diff);
        
        if (magnitude_diff > tolerance) {
            differences_count++;
        }
        
        if (magnitude_diff > max_diff) {
            max_diff = magnitude_diff;
            max_diff_idx = i;
        }
        
        std::cout << i << "\t\t(" << sycl_output[i].real() << ", " << sycl_output[i].imag() << ")"
                  << "\t(" << rtl_output[i].real() << ", " << rtl_output[i].imag() << ")"
                  << "\t(" << diff.real() << ", " << diff.imag() << ")" << std::endl;
    }
    
    std::cout << "\n=== Summary ===\n";
    std::cout << "Total points compared: " << sycl_output.size() << std::endl;
    std::cout << "Points with differences > " << tolerance << ": " << differences_count << std::endl;
    std::cout << "Maximum difference: " << max_diff << " at index " << max_diff_idx << std::endl;
    
    if (differences_count == 0) {
        std::cout << "✓ Outputs match within tolerance!" << std::endl;
    } else {
        std::cout << "⚠ Outputs differ significantly" << std::endl;
    }
}

int main() {
    const size_t num_test_structs = kSize;  // Use full size like main.cpp
    
    std::cout << "=== FFT Comparison Test ===\n";
    std::cout << "Comparing SYCL IFFT vs RTL FFT outputs\n";
    std::cout << "Using first " << num_test_structs << " input structs\n\n";
    
    // Convert RTL input data to complex array for SYCL
    auto complex_input = convert_to_complex_array(fft_test_input, num_test_structs);
    
    std::cout << "Input size: " << complex_input.size() << " complex numbers\n";
    std::cout << "First 8 input values:\n";
    for (size_t i = 0; i < 8; i++) {
        std::cout << "  [" << i << "]: (" << complex_input[i].real() << ", " << complex_input[i].imag() << ")\n";
    }
    
    // Run SYCL IFFT
    std::cout << "\nRunning SYCL IFFT...\n";
    auto sycl_output = run_sycl_ifft(complex_input);
    
    // Run RTL FFT
    std::cout << "Running RTL FFT...\n";
    auto rtl_output_structs = run_rtl_fft(fft_test_input, num_test_structs);
    auto rtl_output = convert_rtl_output_to_complex(rtl_output_structs.data(), num_test_structs);
    
    // Write outputs to files
    std::cout << "\nWriting outputs to files...\n";
    
    // Write SYCL output to file
    std::ofstream sycl_file("SYCL_OUTPUT");
    if (sycl_file.is_open()) {
        sycl_file << std::fixed << std::setprecision(12);
        for (size_t i = 0; i < sycl_output.size(); i++) {
            sycl_file << i << "\t" << sycl_output[i].real() << "\t" << sycl_output[i].imag() << std::endl;
        }
        sycl_file.close();
        std::cout << "SYCL output written to SYCL_OUTPUT\n";
    } else {
        std::cerr << "Error: Could not open SYCL_OUTPUT file for writing\n";
    }
    
    // Write RTL output to file
    std::ofstream rtl_file("RTL_OUTPUT");
    if (rtl_file.is_open()) {
        rtl_file << std::fixed << std::setprecision(12);
        for (size_t i = 0; i < rtl_output.size(); i++) {
            rtl_file << i << "\t" << rtl_output[i].real() << "\t" << rtl_output[i].imag() << std::endl;
        }
        rtl_file.close();
        std::cout << "RTL output written to RTL_OUTPUT\n";
    } else {
        std::cerr << "Error: Could not open RTL_OUTPUT file for writing\n";
    }
    
    // Compare outputs
    compare_outputs(sycl_output, rtl_output);
    
    return 0;
}