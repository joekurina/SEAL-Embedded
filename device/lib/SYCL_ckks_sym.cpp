#include "SYCL_ckks_sym.h"
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

// Include the pipe definitions
#include "SYCL_pipes.h"

// Include the SYCL kernel headers
#include "SYCL_ifft.h"
#include "SYCL_poly_mult_neg.h"
#include "SYCL_poly_add.h"
#include "SYCL_scale_and_reduce.h"
#include "SYCL_lanes.h"

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

// Host-side packing helpers to normalize scalar arrays into 4-lane structs.
template <typename Struct4, typename Scalar>
static inline void pack4(const Scalar* src, size_t n, std::vector<Struct4>& dst) {
    dst.resize(n / 4);
    for (size_t i = 0; i < n; i += 4) {
        Struct4 block{};
        block.element0 = src[i + 0];
        block.element1 = src[i + 1];
        block.element2 = src[i + 2];
        block.element3 = src[i + 3];
        dst[i / 4] = block;
    }
}

template <typename Struct4, typename Scalar>
static inline void unpack4(const std::vector<Struct4>& src, size_t n, Scalar* dst) {
    for (size_t i = 0; i < n; i += 4) {
        const Struct4& block = src[i / 4];
        dst[i + 0] = block.element0;
        dst[i + 1] = block.element1;
        dst[i + 2] = block.element2;
        dst[i + 3] = block.element3;
    }
}

// Forward declare kernel names
class IFFTKernel;
class RTLNTTKernel_A_Input;
class RTLNTTKernel_A;
class RTLNTTKernel_A_Output;
class RTLNTTKernel_B_Input;
class RTLNTTKernel_B;
class RTLNTTKernel_B_Output;
class PolyMultNegNTTKernel;
class PolyAddModKernel;
class ScaleAndReduceKernel;

// Modulus selector function
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

// Forward declare pipeline function
void pipeline(
    queue q,
    size_t n,
    size_t logn,
    double scale,
    uint32_t mod_value,
    uint32_t root,
    const uint32_t* const_ratio,
    buffer<encoding_buffer_input, 1>& encoding_buf,
    buffer<i8x4_input, 1>& error_samples_buf,
    buffer<u32x4_input, 1>& ntt_pte_buf,
    buffer<u32x4_input, 1>& c0_s_buf,
    buffer<u32x4_input, 1>& c1_buf,
    buffer<u32x4_input, 1>& s_save_buf
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
     if (n % 4 != 0) {
          std::cerr << "[SYCL_combined_encrypt] polynomial degree must be divisible by 4 for 4-lane normalization\n";
          std::exit(1);
     }
    
    // Calculate the NTT root
    uint32_t root = NTT_root(n, mod_value);

    // Normalize inputs into 4-lane blocks (host-side only).
    std::vector<encoding_buffer_input> encoding_blocks;
    std::vector<u32x4_input> s_blocks;
    std::vector<u32x4_input> a_blocks;
    std::vector<i8x4_input> error_blocks;
    std::vector<u32x4_input> ntt_pte_blocks;
    std::vector<u32x4_input> c0_s_blocks;
    std::vector<u32x4_input> c1_blocks;
    std::vector<u32x4_input> s_save_blocks;

    pack4(encoding_buffer, n, encoding_blocks);
    pack4(expanded_s, n, s_blocks);
    pack4(uniform_poly, n, a_blocks);
    pack4(error_samples, n, error_blocks);
    pack4(ntt_pte, n, ntt_pte_blocks);

    // Initialize working copies
    c0_s_blocks = s_blocks; // start from expanded_s
    c1_blocks = a_blocks;   // uniform_poly
    if (s_save) {
        pack4(s_save, n, s_save_blocks);
    } else {
        s_save_blocks.resize(n / 4); // provide valid storage even when not requested
    }

    // If requested, save original c1.
    if (c1_save != nullptr) {
        unpack4(a_blocks, n, c1_save);
    }

    // Create SYCL buffers that associate host memory pointers with device-accessible objects.
    // Data will be implicitly managed (copied to/from device) by the SYCL runtime as needed by kernel accessors.

    // Buffer for the input complex-encoded plaintext values. Read by IFFTKernel.
    // IFFT now consumes packed 4-lane structs
    buffer<encoding_buffer_input, 1> encoding_buf(encoding_blocks.data(), range(n / 4));

    // Buffer for the input error samples (packed 4-lane).
    buffer<i8x4_input, 1> error_samples_buf(error_blocks.data(), range(n / 4));

    // Buffer used for intermediate storage and NTT of plaintext+error (packed).
    buffer<u32x4_input, 1> ntt_pte_buf(ntt_pte_blocks.data(), range(n / 4));

    // Main working buffer for ciphertext component c0 (packed).
    buffer<u32x4_input, 1> c0_s_buf(c0_s_blocks.data(), range(n / 4));

    // Input buffer holding the uniform polynomial 'a' (ciphertext component c1) (packed).
    buffer<u32x4_input, 1> c1_buf(c1_blocks.data(), range(n / 4));

    // Buffer for optionally saving the NTT(s) state (packed). Always back by valid storage.
    buffer<u32x4_input, 1> s_save_buf(s_save_blocks.data(), range(n / 4));

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

    // Results are now in packed vectors; unpack back to caller buffers.
    unpack4(c0_s_blocks, n, c0_s);
    unpack4(ntt_pte_blocks, n, ntt_pte);
    unpack4(c1_blocks, n, c1);
    if (s_save && !s_save_blocks.empty()) {
        unpack4(s_save_blocks, n, s_save);
    }
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
    buffer<encoding_buffer_input, 1>& encoding_buf,  // Input buffer: Complex-encoded plaintext values (packed 4-lane).
    buffer<i8x4_input, 1>& error_samples_buf,        // Input buffer: Noise/error samples (packed 4-lane).
    buffer<u32x4_input, 1>& ntt_pte_buf,             // Buffer for NTT(plaintext + error) (packed 4-lane).
    buffer<u32x4_input, 1>& c0_s_buf,                // Main work buffer (packed 4-lane).
    buffer<u32x4_input, 1>& c1_buf,                  // Input buffer: Uniform polynomial 'a' (packed 4-lane).
    buffer<u32x4_input, 1>& s_save_buf               // Output buffer: Destination for saving the NTT(s) state (packed 4-lane).
) {
    uint8_t modulus_selector = get_rtl_modulus_selector(mod_value);
    std::cout << "[Pipeline] Using modulus selector: " << static_cast<int>(modulus_selector) << " for modulus " << mod_value << std::endl;
    try {

        // Submit consumers first to avoid pipe deadlocks; track only finite kernels for waiting.
        std::vector<event> finite_events;

        // RTL NTT A Output: reads from NTTAOutputPipe, writes to NTTToPolyMultNegPipe (finite)
        finite_events.push_back(q.submit([&](handler &h) {
            RTLNTTKernel_A_Output kernel(n, s_save_buf);
            kernel(h);
        }));

        // RTL NTT A Main: infinite loop; do not wait on this event
        q.submit([&](handler &h) {
            RTLNTTKernel_A kernel{};
            kernel(h);
        });

        // RTL NTT A Input: feeds secret key into NTT A (finite)
        finite_events.push_back(q.submit([&](handler &h) {
            RTLNTTKernel_A_Input(n, modulus_selector, c0_s_buf)(h);
        }));

        // PolyMultNegNTTKernel (finite)
        finite_events.push_back(q.submit([&](handler &h) {
            PolyMultNegNTTKernel(n, mod_value, const_ratio, c1_buf)(h);
        }));

        // IFFTKernel (finite)
        finite_events.push_back(q.submit([&](handler &h) {
            IFFTKernel(n, logn, encoding_buf)(h);
        }));

        // ScaleAndReduceKernel (finite)
        finite_events.push_back(q.submit([&](handler &h) {
            ScaleAndReduceKernel(n, scale, mod_value, const_ratio, error_samples_buf)(h);
        }));

        // RTL NTT B Output (finite)
        finite_events.push_back(q.submit([&](handler &h) {
            RTLNTTKernel_B_Output kernel(n, ntt_pte_buf);
            kernel(h);
        }));

        // RTL NTT B Main: infinite loop; do not wait on this event
        q.submit([&](handler &h) {
            RTLNTTKernel_B kernel{};
            kernel(h);
        });

        // RTL NTT B Input (finite)
        finite_events.push_back(q.submit([&](handler &h) {
            RTLNTTKernel_B_Input kernel(n, modulus_selector);
            kernel(h);
        }));

        // PolyAddModKernel (finite, final write-out of c0_s_buf)
        finite_events.push_back(q.submit([&](handler &h) {
            PolyAddModKernel(n, mod_value, c0_s_buf)(h);
        }));

        // Wait only on finite kernels to ensure outputs are ready, while not blocking on infinite RTL mains.
        for (auto &ev : finite_events) {
            ev.wait();
        }

    } catch (std::exception const &e) { // Catch other standard exceptions
        std::cout << "[Pipeline] STANDARD EXCEPTION CAUGHT!" << std::endl;
        std::cerr << "Caught a standard exception in pipeline: "
                  << e.what() << std::endl;
        std::exit(1);
    }
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