#include <stddef.h>
#include <stdint.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include "SYCL_ntt_a.h"
#include "SYCL_RTL_ntt_a_input.h"
#include "SYCL_RTL_ntt_a.h"
#include "SYCL_RTL_ntt_a_output.h"
#include "the_nwc_4k_ntt_sycl.hpp"

using namespace sycl;

class NTTKernel_A;
class RTLNTTKernel_A_Input;
class RTLNTTKernel_A;
class RTLNTTKernel_A_Output;

uint32_t modulus = 1053818881;
uint32_t root = 503422;
uint32_t const_ratio[2] = { 324793530, 4 };
size_t n = 4096;
size_t logn = 12;

uint32_t input_data[4096];

const char* input_file_A = "NTT_A_INPUT.txt";
const char* input_file_B = "NTT_B_INPUT.txt";

// Utility function to map SEAL-Embedded modulus values to RTL modulus selector
inline uint8_t get_rtl_modulus_selector(uint32_t mod_value) {
    switch(mod_value) {
        case 134012929:  return 0;  // root = 7470
        case 134111233:  return 1;  // root = 3856
        case 134176769:  return 2;  // root = 24149
        case 1053818881: return 3;  // root = 503422
        case 1054015489: return 4;  // root = 16768
        case 1054212097: return 5;  // root = 7305
        default:          return 0;  // fallback to first modulus
    }
}

// Function to read data from file into array
void read_input_data(const char* filename, uint32_t* array, size_t size) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }
    for (size_t i = 0; i < size; ++i) {
        file >> array[i];
    }
    file.close();
}

int main() {
    // Read the input data
    read_input_data(input_file_B, input_data, n);

    std::cout << "Data loaded successfully." << std::endl;
    
    // Create SYCL buffers for input and output
    buffer<uint32_t, 1> input_buffer(input_data, range<1>(n));
    buffer<uint32_t, 1> SYCL_output_buffer{range<1>(n)};
    buffer<uint32_t, 1> RTL_output_buffer{range<1>(n)};

    // Create queue
#if FPGA_HARDWARE
    auto selector = ext::intel::fpga_selector_v;
#else
    auto selector = ext::intel::fpga_emulator_selector_v;
#endif
    queue q{selector, property::queue::enable_profiling()};
    std::cout << "Queue created successfully." << std::endl;

    try {

        // Submit the standard SYCL NTT A kernel
        q.submit([&](handler &h) {
            NTTKernel_A kernel(n, logn, modulus, root, const_ratio, input_buffer, SYCL_output_buffer);
            kernel(h);
        });
        std::cout << "NTTKernel_A submitted." << std::endl;

        // Submit the RTL NTT OUTPUT kernel: reads from NTTAOutputPipe, writes to RTL_output_buffer
        q.submit([&](handler &h) {
            RTLNTTKernel_A_Output kernel(RTL_output_buffer);
            kernel(h);
        });
        std::cout << "RTLNTTKernel_A_Output submitted." << std::endl;

        // Submit the RTL NTT A Main kernel: reads from NTTAInputPipe, writes to NTTAOutputPipe
        q.submit([&](handler &h) {
            RTLNTTKernel_A kernel;
            kernel(h);
        });
        std::cout << "RTLNTTKernel_A submitted." << std::endl;

        // Submit the RTL NTT A Input kernel: reads from input_buffer, writes to NTTAInputPipe
        q.submit([&](handler &h) {
            RTLNTTKernel_A_Input kernel(input_buffer);
            kernel(h);
        });
        std::cout << "RTLNTTKernel_A_Input submitted." << std::endl;

    } catch (std::exception const &e) { // Catch other standard exceptions
        std::cout << "[Pipeline] STANDARD EXCEPTION CAUGHT!" << std::endl;
        std::cerr << "Caught a standard exception in pipeline: "
                  << e.what() << std::endl;
        std::exit(1);
    }

    // Results are now in SYCL_output_buffer and RTL_output_buffer host pointers (due to buffer destruction sync)
    {
        std::ofstream file("SYCL_NTT_OUTPUT.txt");
        auto host_acc = SYCL_output_buffer.get_host_access(sycl::read_only);
        for (size_t i = 0; i < n; ++i) {
            file << host_acc[i] << std::endl;
        }
    }
    {
        std::ofstream file("RTL_NTT_OUTPUT.txt");
        auto host_acc = RTL_output_buffer.get_host_access(sycl::read_only);
        for (size_t i = 0; i < n; ++i) {
            file << host_acc[i] << std::endl;
        }
    }

    return 0;
}



