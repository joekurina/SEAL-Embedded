#pragma once

#include "SYCL_ckks_sym.h"
#include "SYCL_pipes.h"
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

// Exit Kernel that reads from pipes and writes to buffers
class ExitKernel {
private:
    size_t n;
    mutable sycl::buffer<int64_t, 1> pt_with_error_acc;

public:
    ExitKernel(size_t n_val,
               sycl::buffer<int64_t, 1>& pt_with_error_buf)
        : n(n_val), 
          pt_with_error_acc(pt_with_error_buf) {}
    
    void operator()(sycl::handler& h) const {
        // Get access to the buffer
        auto pt_with_error = pt_with_error_acc.get_access<sycl::access::mode::write>(h);
        
        // Capture necessary variables
        size_t kernel_n = n;
        
        // Create a stream for debug output
        sycl::stream out(1024, 256, h);
        
        h.single_task([=]() [[intel::kernel_args_restrict]] {
            out << "Exit: Starting to read " << kernel_n << " values from pipe\n" << sycl::flush;
            
            // Process elements in a streaming fashion
            for (size_t i = 0; i < kernel_n; i++) {
                // Read a single value from the pipe
                int64_t value;
                
                // Log first read attempt for debugging
                if (i == 0) {
                    out << "Exit: Attempting to read first value from pipe\n" << sycl::flush;
                }
                
                // Read value from pipe
                value = ScaleAndConvertToExitPipe::read();
                
                // Log successful first read
                if (i == 0) {
                    out << "Exit: Successfully read first value from pipe\n" << sycl::flush;
                }
                
                // Store the value in the output buffer
                pt_with_error[i] = value;
                
                // Print progress at regular intervals
                if (i == 0 || i == kernel_n-1 || i % 1000 == 0) {
                    out << "Exit: Read and stored " << (i+1) << "/" << kernel_n << " values\n" << sycl::flush;
                }
            }
            
            out << "Exit: All values successfully read and stored\n" << sycl::flush;
        });
    }
};