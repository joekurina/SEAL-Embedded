#include "SYCL_pipeline.h"
#include "SYCL_ifft.h"
#include "SYCL_scale_and_convert.h"
#include "SYCL_pipes.h"
#include <iostream>

// Function to perform the pipeline of kernels
void pipeline(
    queue q,
    size_t n,                           
    size_t logn,                        
    double scale,                       
    buffer<std::complex<double>, 1>& encoding_buf,
    buffer<int8_t, 1>& error_samples_buf,
    buffer<int64_t, 1>& pt_with_error_buf
) {
    try {
        // Submit the IFFT kernel
        auto ifft_event = q.submit([&](handler &h) {
            IFFTKernel(n, logn, encoding_buf, error_samples_buf)(h);
        });

        // Submit the ScaleAndConvert kernel
        auto scale_event = q.submit([&](handler &h) {
            // Make the scale kernel depend on the IFFT kernel
            h.depends_on(ifft_event);
            ScaleAndConvertKernel(n, scale, pt_with_error_buf)(h);
        });

        // Wait for all kernels to complete
        scale_event.wait();

        std::cout << "Pipeline execution completed successfully." << std::endl;

    } catch (exception const &e) {
        std::cerr << "Caught a synchronous SYCL exception in pipeline: "
                  << e.what() << std::endl;
        std::exit(1);
    }
}