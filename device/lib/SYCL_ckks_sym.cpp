#include "SYCL_ckks_sym.h"
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

// Include the pipe definitions
#include "SYCL_pipes.h"

// Include the SYCL kernel headers
#include "SYCL_ifft.h"
#include "SYCL_ntt.h"
#include "SYCL_poly_mult_neg.h"
#include "SYCL_poly_add.h"
#include "SYCL_scale_and_reduce.h"

#include <iostream>
#include <vector>

using namespace sycl;

// Forward declare kernel names
class IFFTKernel;
class NTTKernel_1;
class NTTKernel_2;
class PolyMultNegNTTKernel;
class PolyAddModKernel;
class ScaleAndReduceKernel;

// Forward declare pipeline function
void pipeline(
    queue q,
    size_t n,
    size_t logn,
    double scale,
    uint32_t mod_value,
    const uint32_t* const_ratio,
    buffer<std::complex<double>, 1>& encoding_buf,
    buffer<int8_t, 1>& error_samples_buf,
    buffer<uint32_t, 1>& ntt_pte_buf,
    buffer<uint32_t, 1>& c0_s_buf,
    buffer<uint32_t, 1>& c1_buf,
    buffer<uint32_t, 1>& s_save_buf
);

// Implementation of the C-compatible function (Main host interface)
extern "C" void SYCL_combined_encrypt(
    // --- CKKS Parameters ---
    size_t n,                       // Polynomial degree.
    size_t logn,                    // Base-2 logarithm of the polynomial degree.
    double scale,                   // CKKS scaling factor.

    // --- Modulus Information ---
    uint32_t mod_value,             // Current modulus prime (q).
    const uint32_t* const_ratio,    // Precomputed constant ratio for Barrett reduction modulo q.

    // --- Input Data Buffers (Host Pointers) ---
    complex_double* encoding_buffer, // Input: Buffer holding complex-encoded plaintext values (fed to IFFTKernel).
    uint32_t* expanded_s,           // Input: Expanded secret key polynomial 's'.
    uint32_t* uniform_poly,         // Input: Uniformly sampled polynomial 'a' (becomes ciphertext component c1).
    int8_t* error_samples,          // Input: Buffer holding pre-sampled noise/error values.
    int64_t* pt_with_error,         // Input (UNUSED): Previously held plaintext+error; now unused by the SYCL pipeline.

    // --- Input/Output & Scratch Buffers (Host Pointers) ---
    uint32_t* ntt_pte,              // Intermediate/Scratch: Output buffer for ScaleAndReduce, then Input/Output buffer for NTTKernel_1. Holds NTT(plaintext + error).
    uint32_t* c0_s,                 // Input/Output: Starts with expanded_s, used for NTT(s), then -(NTT(s)*c1), finally holds the resulting ciphertext component c0.
    uint32_t* c1,                   // Output: Destination for the uniform polynomial 'a', becomes ciphertext component c1.

    // --- Output Buffers for Testing (Host Pointers) ---
    uint32_t* s_save,               // Output: Destination buffer for saving the NTT(s) state from NTTKernel_2. NULL if not needed.
    uint32_t* c1_save               // Output: Destination buffer for saving the original uniform polynomial 'a'. NULL if not needed.
) {
    
    // Copy the pre-generated uniform polynomial 'a' into the host memory buffer 'c1'.
    // This buffer 'c1' will be associated with a SYCL buffer and also represents
    // the second component of the final ciphertext (c1 = a).
    std::memcpy(c1, uniform_poly, n * sizeof(uint32_t));

    // Check if the save buffer 'c1_save' was provided by the caller.
    if (c1_save != nullptr) {
        // If provided, copy the original uniform polynomial 'a' into 'c1_save'
        // for testing, before 'c1' might be used otherwise.
        std::memcpy(c1_save, uniform_poly, n * sizeof(uint32_t));
    }

    // Copy the pre-generated expanded secret key 's' into the host memory buffer 'c0_s'.
    // This buffer 'c0_s' will be associated with a SYCL buffer and used as the
    // primary working buffer for calculating the first ciphertext component c0 = [-a*s + m + e].
    std::memcpy(c0_s, expanded_s, n * sizeof(uint32_t));

    // Create SYCL buffers that associate host memory pointers with device-accessible objects.
    // Data will be implicitly managed (copied to/from device) by the SYCL runtime as needed by kernel accessors.

    // Buffer for the input complex-encoded plaintext values. Read by IFFTKernel.
    buffer<std::complex<double>, 1> encoding_buf(encoding_buffer, range(n));

    // Buffer for the input error samples. Read by IFFTKernel.
    buffer<int8_t, 1> error_samples_buf(error_samples, range(n));

    // Buffer used for intermediate storage and NTT of plaintext+error.
    // Written by ScaleAndReduceKernel, Read/Written by NTTKernel_1, Read by PolyAddModKernel.
    buffer<uint32_t, 1> ntt_pte_buf(ntt_pte, range(n));

    // Main working buffer for ciphertext component c0.
    // Initialized with expanded_s, Read/Written by NTTKernel_2, Read/Written by PolyMultNegNTTKernel,
    // Read/Written by PolyAddModKernel. Contains final c0 result at the end.
    buffer<uint32_t, 1> c0_s_buf(c0_s, range(n));

    // Input buffer holding the uniform polynomial 'a' (ciphertext component c1).
    // Read by PolyMultNegNTTKernel.
    buffer<uint32_t, 1> c1_buf(c1, range(n));

    // Buffer for optionally saving the NTT(s) state. Initialized to null/invalid range.
    buffer<uint32_t, 1> s_save_buf{nullptr, range(n)};

    // Check if the host requested saving the NTT(s) state (s_save pointer is not NULL).
    if (s_save != nullptr) {
         // If requested, create a valid SYCL buffer associated with the host s_save pointer.
         // This buffer will be written to by NTTKernel_2.
         s_save_buf = buffer<uint32_t, 1>(s_save, range(n));
    }

    // Create queue
#if FPGA_HARDWARE
    auto selector = ext::intel::fpga_selector_v;
#else
    auto selector = ext::intel::fpga_emulator_selector_v;
#endif
    queue q{selector, property::queue::enable_profiling()};

    // Execute the full pipeline
    pipeline(
        q,
        n,
        logn,
        scale,
        mod_value,
        const_ratio,
        encoding_buf,
        error_samples_buf,
        ntt_pte_buf,
        c0_s_buf,
        c1_buf,
        s_save_buf
    );

    // Results are now in c0_s and s_save host pointers (due to buffer destruction sync)
}

// Integrated pipeline function with modified NTTKernel_2 call
void pipeline(
    queue q,                                        // The SYCL queue for submitting kernels.
    size_t n,                                       // The polynomial degree.
    size_t logn,                                    // Base-2 logarithm of the polynomial degree.
    double scale,                                   // The CKKS scaling factor.
    uint32_t mod_value,                             // The current modulus prime (q).
    const uint32_t* const_ratio,                    // Precomputed constant ratio for Barrett reduction modulo q.
    buffer<std::complex<double>, 1>& encoding_buf,  // Input buffer: Complex-encoded plaintext values.
    buffer<int8_t, 1>& error_samples_buf,           // Input buffer: Noise/error samples.
    buffer<uint32_t, 1>& ntt_pte_buf,               // Intermediate buffer: Output of ScaleReduce, Input/Output of NTT1. Holds NTT(plaintext + error).
    buffer<uint32_t, 1>& c0_s_buf,                  // Main work buffer: Input is expanded_s, intermediate results include NTT(s) and -(NTT(s)*c1), final output is ciphertext component c0.
    buffer<uint32_t, 1>& c1_buf,                    // Input buffer: Uniform polynomial 'a' (ciphertext component c1). Read by PolyMultNeg.
    buffer<uint32_t, 1>& s_save_buf                 // Output buffer: Destination for saving the NTT(s) state from NTTKernel_2 if requested.
) {
    std::cout << "[Pipeline] Starting Full Integration (NTT2 Dual Output)..." << std::endl;
    try {

        // --- Kernels that can start immediately ---

        // Submit IFFTKernel
        sycl::event ifft_event = q.submit([&](handler &h) {
            // Assuming IFFTKernel call logic was here
            IFFTKernel(n, logn, encoding_buf, error_samples_buf)(h);
        });
        std::cout << "[Pipeline] Submitted IFFTKernel." << std::endl; // Print moved after submit for clarity

        // Submit MODIFIED NTTKernel_2 (Operates on c0_s_buf AND writes to s_save_buf)
        std::cout << "[Pipeline] Submitting NTTKernel_2 (Dual Output)..." << std::endl;
        sycl::event ntt2_event = q.submit([&](handler &h) {
            // Pass both primary buffer and save buffer
            NTTKernel_2(n, logn, mod_value, const_ratio, c0_s_buf, s_save_buf)(h);
        });
        std::cout << "[Pipeline] Submitted NTTKernel_2." << std::endl;

        // --- Kernels dependent on IFFT ---
        std::cout << "[Pipeline] Submitting ScaleAndReduceKernel..." << std::endl; // Print moved before submit
        sycl::event scale_reduce_event = q.submit([&](handler &h) {
             h.depends_on(ifft_event);
             // Assuming ScaleAndReduceKernel call logic was here
             ScaleAndReduceKernel(n, scale, mod_value, const_ratio, ntt_pte_buf)(h);
        });
        std::cout << "[Pipeline] Submitted ScaleAndReduceKernel." << std::endl; // Print moved after submit

        // --- Kernels dependent on ScaleAndReduce ---
        std::cout << "[Pipeline] Submitting NTTKernel_1..." << std::endl; // Print moved before submit
        sycl::event ntt1_event = q.submit([&](handler &h) {
             h.depends_on(scale_reduce_event);
             // Assuming NTTKernel_1 call logic was here
             NTTKernel_1(n, logn, mod_value, const_ratio, ntt_pte_buf)(h);
        });
        std::cout << "[Pipeline] Submitted NTTKernel_1." << std::endl; // Print moved after submit

        // Submit PolyMultNegNTTKernel
        // Depends on NTT2 completing its write to c0_s_buf
        std::cout << "[Pipeline] Submitting PolyMultNegNTTKernel..." << std::endl;
        sycl::event mult_neg_event = q.submit([&](handler &h) {
            h.depends_on(ntt2_event); // Depends on NTT2 completion
            PolyMultNegNTTKernel(n, mod_value, const_ratio, c0_s_buf, c1_buf)(h);
        });
        std::cout << "[Pipeline] Submitted PolyMultNegNTTKernel." << std::endl;

        // --- Final Kernel dependent on NTT1 and MultNeg ---
        std::cout << "[Pipeline] Submitting PolyAddModKernel..." << std::endl;
        sycl::event add_event = q.submit([&](handler &h) {
            // Depends on NTT1 and MultNeg completing
            h.depends_on({ntt1_event, mult_neg_event});
            PolyAddModKernel(n, mod_value, c0_s_buf, ntt_pte_buf)(h);
        });
        std::cout << "[Pipeline] Submitted PolyAddModKernel." << std::endl;

        // Wait for the final PolyAddModKernel kernel to complete
        add_event.wait();

        std::cout << "[Pipeline] Full pipeline execution completed." << std::endl;

    } catch (exception const &e) {
         std::cout << "[Pipeline] EXCEPTION CAUGHT!" << std::endl; // Indented catch block
         std::cerr << "Caught a synchronous SYCL exception in pipeline: "
                   << e.what() << std::endl;
         std::exit(1); // Consider if exit is appropriate or if error should propagate
    }
    std::cout << "[Pipeline] Exiting." << std::endl;
}