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
        
        h.single_task([=]() [[intel::kernel_args_restrict]] {
            // Read results from pipe and write to buffer
            for (size_t i = 0; i < kernel_n; i++) {
                pt_with_error[i] = ScaleAndConvertToExitPipe::read();
            }
        });
    }
};