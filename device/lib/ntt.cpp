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

void ntt_inpl(const Parms *parms, const ZZ *ntt_roots, ZZ *vec)
{
    se_assert(parms && parms->curr_modulus && vec);
    SE_UNUSED(ntt_roots);  // Not used in OTF implementation
    
    size_t n = parms->coeff_count;
    size_t logn = parms->logn;
    Modulus *mod = parms->curr_modulus;
    
    // Get root for NTT
    ZZ root = get_ntt_root(n, mod->value);
    
    // Perform NTT in scrambled order
    size_t h = 1;
    size_t tt = n / 2;

    for (size_t i = 0; i < logn; i++, h *= 2, tt /= 2)  // rounds
    {
        for (size_t j = 0, kstart = 0; j < h; j++, kstart += 2 * tt)  // groups
        {
            ZZ power = h + j;
            
            // Inline exponentiation
            ZZ s;
            if (power == 0) {
                s = 1;
            }
            else if (power == (1 << (logn - 1))) {
                s = root;
            }
            else {
                ZZ current_power = root;
                ZZ product = 0;
                ZZ result = 1;
                size_t shift_count = logn - 1;
                
                while (true) {
                    if (power & (ZZ)(1 << shift_count)) {
                        product = mul_mod(current_power, result, mod);
                        result = product;
                    }
                    
                    power &= (ZZ)(~(1 << shift_count));
                    if (power == 0) {
                        s = result;
                        break;
                    }
                    
                    product = mul_mod(current_power, current_power, mod);
                    current_power = product;
                    shift_count--;
                }
            }
            
            for (size_t k = kstart; k < (kstart + tt); k++)  // pairs
            {
                ZZ u = vec[k];
                ZZ v = mul_mod(vec[k + tt], s, mod);
                vec[k] = add_mod(u, v, mod);
                vec[k + tt] = sub_mod(u, v, mod);
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
                    print_zz("Modulus value", q);
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
                    print_zz("Modulus value", q);
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
                    print_zz("Modulus value", q);
                    exit(1);
                }
            }
            break;
        default: {
            printf("Error! Need first power of root for ntt\n");
            print_zz("Modulus value", q);
            exit(1);
        }
    }
    return root;
}