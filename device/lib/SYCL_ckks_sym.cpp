#include "SYCL_ckks_sym.h"
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

// Include the pipe definitions
#include "SYCL_pipes.h"

// Include the SYCL kernel headers
#include "SYCL_ifft.h"
#include "SYCL_ntt_a.h"
#include "SYCL_ntt_b.h"
#include "SYCL_poly_mult_neg.h"
#include "SYCL_poly_add.h"
#include "SYCL_scale_and_reduce.h"

// Include RTL NTT kernel headers
#include "SYCL_RTL_ntt_a_input.h"
#include "SYCL_RTL_ntt_a.h"
#include "SYCL_RTL_ntt_a_output.h"
#include "SYCL_RTL_ntt_b_input.h"
#include "SYCL_RTL_ntt_b.h"
#include "SYCL_RTL_ntt_b_output.h"

#include <iostream>
#include <vector>

using namespace sycl;

// Forward declare kernel names
class IFFTKernel;
class NTTKernel_A;
class NTTKernel_B;
class RTLNTTKernel_A_Input;
class RTLNTTKernel_A;
class RTLNTTKernel_A_Output;
class RTLNTTKernel_B_Input;
class RTLNTTKernel_B;
class RTLNTTKernel_B_Output;
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
    uint32_t root,
    const uint32_t* const_ratio,
    buffer<std::complex<double>, 1>& encoding_buf,
    buffer<int8_t, 1>& error_samples_buf,
    buffer<uint32_t, 1>& ntt_pte_buf,
    buffer<uint32_t, 1>& c0_s_buf,
    buffer<uint32_t, 1>& c1_buf,
    buffer<uint32_t, 1>& s_save_buf
);
// Forward declare NTT root calculation function
uint32_t NTT_root(size_t n, uint32_t mod_val);

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
    
    // Calculate the NTT root
    uint32_t root = NTT_root(n, mod_value);

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
        root,
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
    uint32_t mod_value,                             // The modulus value (q).
    uint32_t root,                                  // The NTT root for the polynomial ring.
    const uint32_t* const_ratio,                    // Precomputed constant ratio for Barrett reduction modulo q.
    buffer<std::complex<double>, 1>& encoding_buf,  // Input buffer: Complex-encoded plaintext values.
    buffer<int8_t, 1>& error_samples_buf,           // Input buffer: Noise/error samples.
    buffer<uint32_t, 1>& ntt_pte_buf,               // MODIFIED ROLE: Now primarily the OUTPUT buffer for NTTKernel_1. Holds NTT(plaintext + error).
    buffer<uint32_t, 1>& c0_s_buf,                  // Main work buffer: Input is expanded_s, intermediate results include NTT(s) and -(NTT(s)*c1), final output is ciphertext component c0.
    buffer<uint32_t, 1>& c1_buf,                    // Input buffer: Uniform polynomial 'a' (ciphertext component c1). Read by PolyMultNeg.
    buffer<uint32_t, 1>& s_save_buf                 // Output buffer: Destination for saving the NTT(s) state from NTTKernel_2 if requested.
) {
    try {

        // Submit consumers first to avoid pipe deadlocks
        // RTL NTT A Output: reads from NTTAOutputPipe, writes to NTTToPolyMultNegPipe
        q.submit([&](handler &h) {
            RTLNTTKernel_A_Output kernel(n)(h);
        });

        // Submit PolyMultNegNTTKernel (reads from NTTToPolyMultNegPipe)
        q.submit([&](handler &h) {
            PolyMultNegNTTKernel(n, mod_value, const_ratio, c1_buf)(h);
        });

        // Submit RTL NTT B Output: reads from NTTBOutputPipe, writes to NTTToAddModPipe and ntt_pte_buf
        q.submit([&](handler &h) {
            RTLNTTKernel_B_Output kernel(n, ntt_pte_buf)(h);
        });

        // PolyAddModKernel reads from pipes written by PolyMultNeg and RTL B output
        q.submit([&](handler &h) {
            PolyAddModKernel(n, mod_value, c0_s_buf)(h);
        });

        // Submit IFFTKernel (writes to pipes read by ScaleAndReduce)
        q.submit([&](handler &h) {
            IFFTKernel(n, logn, encoding_buf)(h);
        });

        // Submit RTL NTT A Input: reads from c0_s_buf, writes to NTTAInputPipe
        q.submit([&](handler &h) {
            RTLNTTKernel_A_Input(n, mod_value, c0_s_buf, s_save_buf)(h);
        });

        // RTL NTT A Main: reads from NTTAInputPipe, writes to NTTAOutputPipe
        q.submit([&](handler &h) {
            RTLNTTKernel_A kernel(mod_value)(h);
        });

        // ScaleAndReduceKernel reads from IFFT pipes and writes to ScaleReduceToNTTBPipe
        q.submit([&](handler &h) {
            ScaleAndReduceKernel(n, scale, mod_value, const_ratio, error_samples_buf)(h);
        });

        // Submit RTL NTT B Input: reads from ScaleReduceToNTTBPipe, writes to NTTBInputPipe
        q.submit([&](handler &h) {
            RTLNTTKernel_B_Input(n, mod_value)(h);
        });

        // RTL NTT B Main: reads from NTTBInputPipe, writes to NTTBOutputPipe
        q.submit([&](handler &h) {
            RTLNTTKernel_B kernel(mod_value)(h);
        });

    } catch (std::exception const &e) { // Catch other standard exceptions
        std::cout << "[Pipeline] STANDARD EXCEPTION CAUGHT!" << std::endl;
        std::cerr << "Caught a standard exception in pipeline: "
                  << e.what() << std::endl;
        std::exit(1);
    }

    // Wait for all kernels to complete
    q.wait();

} // End of pipeline function

uint32_t NTT_root(std::size_t n, uint32_t mod_val)
{
    uint32_t root = 1;

    switch (n)
    {
        case 4096:
            switch (mod_val)
            {
                case 134012929u: root =  7470;  break;
                case 134111233u: root =  3856;  break;
                case 134176769u: root = 24149;  break;
                case 1053818881u: root = 503422; break;
                case 1054015489u: root = 16768;  break;
                case 1054212097u: root =  7305;  break;
                default:                        /* keep root = 1 */ ;
            }
            break;

        case 8192:
            switch (mod_val)
            {
                case 1053818881u: root = 374229; break;
                case 1054015489u: root = 123363; break;
                case 1054212097u: root =  79941; break;
                case 1055260673u: root =  38869; break;
                case 1056178177u: root = 162146; break;
                case 1056440321u: root =  81884; break;
                default:                        /* keep root = 1 */ ;
            }
            break;

        case 16384:
            switch (mod_val)
            {
                case 1053818881u: root =  13040;  break;
                case 1054015489u: root =    507;  break;
                case 1054212097u: root =   1595;  break;
                case 1055260673u: root =  68507;  break;
                case 1056178177u: root =   3073;  break;
                case 1056440321u: root =   6854;  break;
                case 1058209793u: root =  44467;  break;
                case 1060175873u: root =  16117;  break;
                case 1060700161u: root =  27607;  break;
                case 1060765697u: root = 222391;  break;
                case 1061093377u: root = 105471;  break;
                case 1062469633u: root = 310222;  break;
                case 1062535169u: root =   2005;  break;
                default:                        /* keep root = 1 */ ;
            }
            break;

        default:
            /* keep root = 1 */
            break;
    }

    return root;
} // End of NTT_root function