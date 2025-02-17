//––– Prevent inclusion of the problematic headers –––
// These macros (which should match the include guard names in the C headers)
// cause the headers to be skipped in this translation unit.
#define UTIL_PRINT_H
#define FIPS202_H

#ifndef SE_DISABLE_TESTING_CAPABILITY

// Now include the rest of your project headers. Because the include guards
// for util_print.h and fips202.h are already defined, their contents will not
// be re-included here.
extern "C" {
#include "defines.h"
// (Note: if defines.h itself includes util_print.h or fips202.h, those inclusions
// will be skipped because of the macros above.)
#include "ntt.h"
#include <stdio.h>
#include "defines.h"
#include "fft.h"
#include "parameters.h"
#include "polymodarith.h"
#include "uintmodarith.h"
//#include "util_print.h"
}

// NTT root helper function declaration
static ZZ get_ntt_root(size_t n, ZZ q);

void ntt_roots_initialize(const Parms *parms, ZZ *ntt_roots)
{
    SE_UNUSED(parms);
    SE_UNUSED(ntt_roots);
    return;
}

void ntt_inpl(const Parms *parms, const ZZ *ntt_roots, ZZ *vec)
{
    // This function performs an in-place Number Theoretic Transform (NTT) on the input vector 'vec'.
    // 'parms' contains parameters for the NTT, including the modulus and coefficient count.
    // 'ntt_roots' is not used in this implementation (indicated by the SE_UNUSED macro).

    se_assert(parms && parms->curr_modulus && vec);
    // Asserts that 'parms', 'parms->curr_modulus', and 'vec' are not null pointers.
    SE_UNUSED(ntt_roots);  // Not used in OTF implementation
    // Indicates that 'ntt_roots' is not used in this implementation.

    size_t n = parms->coeff_count;
    // 'n' stores the coefficient count, which is the size of the input vector.
    size_t logn = parms->logn;
    // 'logn' stores the base-2 logarithm of 'n'.
    Modulus *mod = parms->curr_modulus;
    // 'mod' points to the current modulus being used for the NTT.

    // Get root for NTT
    ZZ root = get_ntt_root(n, mod->value);
    // 'root' stores a primitive nth root of unity modulo 'mod->value'.

    // Perform NTT in scrambled order
    size_t h = 1;
    // 'h' represents the size of the sub-arrays being processed. It starts at 1 and doubles in each round.
    size_t tt = n / 2;
    // 'tt' represents half the size of the sub-arrays being processed. It starts at n/2 and halves in each round.

    for (size_t i = 0; i < logn; i++, h *= 2, tt /= 2)  // rounds
    {
        // This outer loop iterates 'logn' times, representing the rounds of the NTT algorithm.
        // In each iteration, 'h' is doubled and 'tt' is halved.
        for (size_t j = 0, kstart = 0; j < h; j++, kstart += 2 * tt)  // groups
        {
            // This inner loop iterates 'h' times, processing groups of elements within the sub-arrays.
            // 'kstart' keeps track of the starting index for each group. It increments by 2*'tt' in each iteration.
            ZZ power = h + j;
            // 'power' determines the power to which the root of unity is raised.

            // Inline exponentiation
            ZZ s;
            // 's' will store the root of unity raised to the power 'power'.
            if (power == 0) {
                s = 1;
            }
            // If 'power' is 0, 's' is set to 1 (root^0 = 1).
            else if (power == (1 << (logn - 1))) {
                s = root;
            }
            // If 'power' is 2^(logn-1), 's' is set to the root itself.
            else {
                ZZ current_power = root;
                ZZ product = 0;
                ZZ result = 1;
                size_t shift_count = logn - 1;
                // This block calculates root^power using the square and multiply algorithm.
                
                while (true) {
                    if (power & (ZZ)(1 << shift_count)) {
                        product = mul_mod(current_power, result, mod);
                        result = product;
                    }
                    // If the current bit of 'power' is 1, multiply 'result' by 'current_power'.
                    
                    power &= (ZZ)(~(1 << shift_count));
                    if (power == 0) {
                        s = result;
                        break;
                    }
                    // Clear the current bit of 'power'. If 'power' is now 0, the exponentiation is complete.
                    
                    product = mul_mod(current_power, current_power, mod);
                    current_power = product;
                    shift_count--;
                }
            }
            
            for (size_t k = kstart; k < (kstart + tt); k++)  // pairs
            {
                // This innermost loop iterates 'tt' times, processing pairs of elements within each group.
                ZZ u = vec[k];
                // 'u' stores the value of the first element in the pair.
                ZZ v = mul_mod(vec[k + tt], s, mod);
                // 'v' stores the value of the second element in the pair, multiplied by the twiddle factor 's'.
                vec[k] = add_mod(u, v, mod);
                // The first element is updated with the sum of 'u' and 'v' modulo 'mod'.
                vec[k + tt] = sub_mod(u, v, mod);
                // The second element is updated with the difference of 'u' and 'v' modulo 'mod'.
            }
        }
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