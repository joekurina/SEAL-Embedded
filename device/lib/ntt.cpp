#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include "defines.h"
#include "ntt.h"
#include <stdio.h>
#include "defines.h"
#include "fft.h"
#include "parameters.h"
#include "polymodarith.h"
#include "uintmodarith.h"
#include "util_print.h"

// NTT root helper function declaration
static ZZ get_ntt_root(size_t n, ZZ q);
    
void ntt_roots_initialize(const Parms *parms, ZZ *ntt_roots)
{
    SE_UNUSED(parms);
    SE_UNUSED(ntt_roots);
    return;
}

extern "C" void ntt_inpl(const Parms *parms, const ZZ *ntt_roots, ZZ *vec) {
    // Check inputs
    se_assert(parms && parms->curr_modulus && vec);
    SE_UNUSED(ntt_roots);  // Not used in this on-the-fly implementation

    size_t n    = parms->coeff_count;
    size_t logn = parms->logn;
    const Modulus *mod = parms->curr_modulus;
    // Get modulus value to be used in the kernel.
    const ZZ mod_val = mod->value;
    
    // Get the primitive root for the NTT.
    const ZZ root = get_ntt_root(n, mod_val);

    // Select the FPGA emulator device.
    auto selector = sycl::ext::intel::fpga_emulator_selector_v;
    try {
        sycl::queue q{selector};
        std::cout << "Running NTT on device: "
                  << q.get_device().get_info<sycl::info::device::name>().c_str()
                  << std::endl;

        // Create a SYCL buffer that wraps the input vector.
        sycl::buffer<ZZ, 1> buf(vec, sycl::range<1>(n));

        q.submit([&](sycl::handler &h) {
            // Create a read-write accessor.
            auto data = buf.get_access<sycl::access::mode::read_write>(h);
            h.single_task<class NTTKernelBuffer>([=]() {
                size_t hsize = 1;
                size_t tt = n / 2;
                // Loop over stages.
                for (size_t i = 0; i < logn; i++, hsize *= 2, tt /= 2) {
                    for (size_t j = 0, kstart = 0; j < hsize; j++, kstart += 2 * tt) {
                        // Compute twiddle factor exponent.
                        ZZ power = hsize + j;
                        ZZ s;
                        if (power == 0) {
                            s = 1;
                        } else if (power == (1 << (logn - 1))) {
                            s = root;
                        } else {
                            // Inline exponentiation: calculate s = root^power mod mod_val.
                            ZZ current_power = root;
                            ZZ result = 1;
                            size_t shift_count = logn - 1;
                            while (true) {
                                if (power & ((ZZ)1 << shift_count)) {
                                    result = mul_mod(current_power, result, mod);
                                }
                                power &= ~((ZZ)1 << shift_count);
                                if (power == 0) {
                                    s = result;
                                    break;
                                }
                                current_power = mul_mod(current_power, current_power, mod);
                                shift_count--;
                            }
                        }
                        // Process each pair in the current group.
                        for (size_t k = kstart; k < (kstart + tt); k++) {
                            ZZ u = data[k];
                            ZZ v = mul_mod(data[k + tt], s, mod);
                            data[k]       = add_mod(u, v, mod);
                            data[k + tt]  = sub_mod(u, v, mod);
                        }
                    }
                }
            });
        });
        q.wait();
    } catch (sycl::exception const &e) {
        std::cerr << "Caught a synchronous SYCL exception in ntt_inpl: "
                  << e.what() << "\n";
        std::exit(1);
    }
}

static ZZ get_ntt_root(size_t n, ZZ q)
{
    ZZ root;
    switch (n)
    {
        case 4096:
            switch (q)
            {
                case 134012929: root = 7470; break;
                case 134111233: root = 3856; break;
                case 134176769: root = 24149; break;
                case 1053818881: root = 503422; break;
                case 1054015489: root = 16768; break;
                case 1054212097: root = 7305; break;
                default: {
                    printf("Error! Need first power of root for ntt, n = 4K\n");
                    //print_zz("Modulus value", q);
                    exit(1);
                }
            }
            break;
        case 8192:
            switch (q)
            {
                case 1053818881: root = 374229; break;
                case 1054015489: root = 123363; break;
                case 1054212097: root = 79941; break;
                case 1055260673: root = 38869; break;
                case 1056178177: root = 162146; break;
                case 1056440321: root = 81884; break;
                default: {
                    printf("Error! Need first power of root for ntt, n = 8K\n");
                    //print_zz("Modulus value", q);
                    exit(1);
                }
            }
            break;
        case 16384:
            switch (q)
            {
                case 1053818881: root = 13040; break;
                case 1054015489: root = 507; break;
                case 1054212097: root = 1595; break;
                case 1055260673: root = 68507; break;
                case 1056178177: root = 3073; break;
                case 1056440321: root = 6854; break;
                case 1058209793: root = 44467; break;
                case 1060175873: root = 16117; break;
                case 1060700161: root = 27607; break;
                case 1060765697: root = 222391; break;
                case 1061093377: root = 105471; break;
                case 1062469633: root = 310222; break;
                case 1062535169: root = 2005; break;
                default: {
                    printf("Error! Need first power of root for ntt, n = 16K\n");
                    //print_zz("Modulus value", q);
                    exit(1);
                }
            }
            break;
        default: {
            printf("Error! Need first power of root for ntt\n");
            //print_zz("Modulus value", q);
            exit(1);
        }
    }
    return root;
}