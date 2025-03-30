#include "SYCL_ckks_sym.h"
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <memory>
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

// Forward declaration of get_ntt_root function
static uint32_t get_ntt_root(size_t n, uint32_t q);

// Define pipe types for transferring data between kernels
// Pipe for complex values from IFFT to Scale & Convert
using ifft_to_scale_pipe = sycl::ext::intel::pipe<class ifft_scale_pipe_id, complex_double, 4096>;

// Pipe for error samples from IFFT to Scale & Convert
using error_to_scale_pipe = sycl::ext::intel::pipe<class error_scale_pipe_id, int8_t, 4096>;

// Pipe for plaintext with error from Scale & Convert to Reduce
using scale_to_reduce_pipe = sycl::ext::intel::pipe<class scale_reduce_pipe_id, int64_t, 4096>;

// IFFT Kernel functor class
class IFFTKernel {
private:
    size_t n;
    size_t logn;
    mutable sycl::buffer<complex_double, 1> encoding_acc;
    mutable sycl::buffer<int8_t, 1> error_samples_acc;

public:
    IFFTKernel(size_t n_val, size_t logn_val, 
                sycl::buffer<complex_double, 1>& encoding_buf,
                sycl::buffer<int8_t, 1>& error_samples_buf)
        : n(n_val), logn(logn_val), encoding_acc(encoding_buf), error_samples_acc(error_samples_buf) {}
    
    void operator()(sycl::handler& h) const {
        // Get access to the buffers
        auto encoding = encoding_acc.get_access<sycl::access::mode::read_write>(h);
        auto error_samples = error_samples_acc.get_access<sycl::access::mode::read>(h);
        
        // Capture necessary variables
        size_t kernel_n = n;
        size_t kernel_logn = logn;
        
        h.single_task([=]() [[intel::kernel_args_restrict]] {
            // Bit-reversal function
            auto bitrev = [](size_t input, size_t numbits) -> size_t {
                size_t t = (((input & 0xaaaa) >> 1) | ((input & 0x5555) << 1));
                t        = (((t & 0xcccc) >> 2) | ((t & 0x3333) << 2));
                t        = (((t & 0xf0f0) >> 4) | ((t & 0x0f0f) << 4));
                t        = (((t & 0xff00) >> 8) | ((t & 0x00ff) << 8));
                return (numbits == 0) ? 0 : (t >> (16 - numbits));
            };
            
            // Root calculation function
            auto calc_root_otf = [](size_t k, size_t m) -> complex_double {
                double angle = 2.0 * M_PI * static_cast<double>(k) / static_cast<double>(m);
                return complex_double(sycl::cos(angle), sycl::sin(angle));
            };
            
            // IFFT implementation
            size_t tt = 1, h = kernel_n / 2;
            
            for (size_t round = 0; round < kernel_logn; round++, tt *= 2, h /= 2) {
                for (size_t j = 0, kstart = 0; j < h; j++, kstart += 2 * tt) {
                    complex_double s;
                    size_t br = bitrev(h + j, kernel_logn);
                    s = std::conj(calc_root_otf(br, kernel_n << 1));
                    
                    for (size_t k = kstart; k < kstart + tt; k++) {
                        complex_double u = encoding[k];
                        complex_double v = encoding[k + tt];
                        encoding[k]      = u + v;
                        encoding[k + tt] = (u - v) * s;
                    }
                }
            }

            // First write all the error samples to the error pipe
            for (size_t i = 0; i < kernel_n; i++) {
                error_to_scale_pipe::write(error_samples[i]);
            }

            // Then write the encoding results to the encoding pipe
            for (size_t i = 0; i < kernel_n; i++) {
                ifft_to_scale_pipe::write(encoding[i]);
            }
        });
    }
};

// Kernel for scaling and conversion to integers, then adding the error samples to the encoded values
class ScaleAndConvertKernel {
private:
    size_t n;
    double scale;

public:
    ScaleAndConvertKernel(size_t n_val, double scale_val)
        : n(n_val), scale(scale_val) {}

    void operator()(sycl::handler& h) const {
        size_t kernel_n = n;
        double kernel_scale = scale;
        
        // Create a stream for debug output
        sycl::stream out(1024, 256, h);
        
        h.single_task([=]() [[intel::kernel_args_restrict]] {
            // Reduce debug output to minimize corruption
            size_t error_samples_read = 0;
            size_t values_written = 0;
            
            // Read all error samples first into a local array
            int8_t error_values[16384];
            for (size_t i = 0; i < kernel_n; i++) {
                error_values[i] = error_to_scale_pipe::read();
                error_samples_read++;
                
                // Only print at the end
                if (i == kernel_n - 1) {
                    out << "Scale: Read all " << kernel_n << " error samples\n";
                }
            }
            
            double n_inv = kernel_scale / static_cast<double>(kernel_n);
            for (size_t i = 0; i < kernel_n; i++) {
                // Read the next complex value from the pipe
                complex_double val = ifft_to_scale_pipe::read();
                
                double real_val = std::real(val);
                double scaled = sycl::round(real_val * n_inv);
                int64_t int_val = static_cast<int64_t>(scaled);
                
                // Write to the next pipe
                int64_t pt_with_error_val = int_val + error_values[i];
                scale_to_reduce_pipe::write(pt_with_error_val);
                values_written++;
                
                // Print progress less frequently
                if (i == 1000 || i == 2000 || i == 3000 || i == kernel_n - 1) {
                    out << "Scale: Written " << values_written << "/" << kernel_n << " values\n";
                }
            }
        });
    }
};

// Kernel for reducing int64_t values from a pipe to their representation in ring Z_q
class PTEReducePipeKernel {
private:
    size_t n;
    uint32_t mod_value;
    const uint32_t* const_ratio;
    mutable sycl::buffer<uint32_t, 1> out_acc;

public:
    PTEReducePipeKernel(size_t n_val, uint32_t mod_val, const uint32_t* const_ratio_val,
                        sycl::buffer<uint32_t, 1>& out_buf)
        : n(n_val), mod_value(mod_val), const_ratio(const_ratio_val), out_acc(out_buf) {}

    void operator()(sycl::handler& h) const {
        auto out = out_acc.get_access<sycl::access::mode::write>(h);
        size_t kernel_n = n;
        uint32_t kernel_mod_val = mod_value;
        const uint32_t* kernel_const_ratio = const_ratio;
        
        sycl::stream debug_stream(1024, 256, h);
        
        h.single_task([=]() [[intel::kernel_args_restrict]] {
            debug_stream << "PTEReduce: Starting to read " << kernel_n << " values from pipe\n";
            
            // First, read all values from the pipe into a local array
            int64_t local_values[16384]; // Assuming max n is 16384
            
            // Read values in smaller chunks to provide progress updates
            const size_t chunk_size = 500;
            for (size_t chunk_start = 0; chunk_start < kernel_n; chunk_start += chunk_size) {
                size_t chunk_end = sycl::min(chunk_start + chunk_size, kernel_n);
                
                debug_stream << "PTEReduce: Reading values " << chunk_start + 1 
                                << " to " << chunk_end << "\n";
                
                // Read this chunk of values
                for (size_t i = chunk_start; i < chunk_end; i++) {
                    local_values[i] = scale_to_reduce_pipe::read();
                }
                
                debug_stream << "PTEReduce: Completed reading values " << chunk_start + 1 
                                << " to " << chunk_end << "\n";
            }
            
            debug_stream << "PTEReduce: Completed reading all " << kernel_n 
                            << " values, starting processing\n";
            
            // Process the values using the proven Barrett reduction logic
            for (size_t i = 0; i < kernel_n; i++) {
                int64_t val = local_values[i];
                
                // Compute absolute value
                uint64_t coeff_abs = (val < 0) ? static_cast<uint64_t>(-val) : static_cast<uint64_t>(val);
                
                // Create mask based on sign (1 if negative, 0 if positive)
                uint32_t mask = static_cast<uint32_t>(val < 0);
                
                // Split 64-bit value into two 32-bit parts for Barrett reduction
                uint32_t coeff_abs_vec[2];
                coeff_abs_vec[0] = static_cast<uint32_t>(coeff_abs & 0xFFFFFFFF);
                coeff_abs_vec[1] = static_cast<uint32_t>((coeff_abs >> 32) & 0xFFFFFFFF);
                
                // Implement Barrett reduction (same as your working code)
                // -- Round 1
                uint32_t right_hw;
                {
                    uint64_t res_temp = (uint64_t)coeff_abs_vec[0] * (uint64_t)kernel_const_ratio[0];
                    right_hw = (uint32_t)((res_temp >> 32) & 0xFFFFFFFF);
                }
                
                uint32_t middle_temp[2];
                {
                    uint64_t res_temp = (uint64_t)coeff_abs_vec[0] * (uint64_t)kernel_const_ratio[1];
                    middle_temp[0] = (uint32_t)(res_temp & 0xFFFFFFFF);
                    middle_temp[1] = (uint32_t)((res_temp >> 32) & 0xFFFFFFFF);
                }
                
                uint32_t middle_lw;
                uint32_t middle_lw_carry;
                {
                    middle_lw = right_hw + middle_temp[0];
                    middle_lw_carry = (uint8_t)(middle_lw < right_hw);
                }
                
                uint32_t middle_hw = middle_temp[1] + middle_lw_carry;
                
                // -- Round 2
                uint32_t middle2_temp[2];
                {
                    uint64_t res_temp = (uint64_t)coeff_abs_vec[1] * (uint64_t)kernel_const_ratio[0];
                    middle2_temp[0] = (uint32_t)(res_temp & 0xFFFFFFFF);
                    middle2_temp[1] = (uint32_t)((res_temp >> 32) & 0xFFFFFFFF);
                }
                
                uint32_t middle2_lw;
                uint32_t middle2_lw_carry;
                {
                    middle2_lw = middle_lw + middle2_temp[0];
                    middle2_lw_carry = (uint8_t)(middle2_lw < middle_lw);
                }
                
                uint32_t middle2_hw = middle2_temp[1] + middle2_lw_carry;
                
                uint32_t tmp = coeff_abs_vec[1] * kernel_const_ratio[1] + middle_hw + middle2_hw;
                
                // -- Barrett subtraction
                tmp = coeff_abs_vec[0] - tmp * kernel_mod_val;
                
                // -- Final reduction if needed
                uint32_t coeff_crt;
                {
                    int32_t is_2q = (int32_t)(tmp >= kernel_mod_val);
                    uint32_t tmp_mask = (uint32_t)(-is_2q);
                    coeff_crt = (uint32_t)(tmp) - (kernel_mod_val & tmp_mask);
                }
                
                // Compute final result based on sign
                uint32_t result = ((kernel_mod_val - coeff_crt) & (-mask)) + (coeff_crt & (mask - 1));
                
                // Store the result
                out[i] = result;
            }
            
            debug_stream << "PTEReduce: Completed processing all " << kernel_n << " values\n";
        });
    }
};

// NTT Kernel functor class
class NTTKernel1 {
private:
    size_t n;
    size_t logn;
    uint32_t mod_value;
    uint32_t root;
    const uint32_t* const_ratio;
    mutable sycl::buffer<uint32_t, 1> vec_acc;

public:
    NTTKernel1(size_t n_val, size_t logn_val, uint32_t mod_val, uint32_t root_val, 
                const uint32_t* const_ratio_val, sycl::buffer<uint32_t, 1>& vec_buf)
        : n(n_val), logn(logn_val), mod_value(mod_val), root(root_val), 
            const_ratio(const_ratio_val), vec_acc(vec_buf) {}
    
    void operator()(sycl::handler& h) const {
        // Get access to the buffer
        auto data = vec_acc.get_access<sycl::access::mode::read_write>(h);
        
        // Capture necessary variables
        size_t kernel_n = n;
        size_t kernel_logn = logn;
        uint32_t kernel_mod_val = mod_value;
        uint32_t kernel_root = root;
        const uint32_t* kernel_const_ratio = const_ratio;
        
        h.single_task([=]() [[intel::kernel_args_restrict]] {
            size_t hsize = 1;
            size_t tt = kernel_n / 2;
            
            // Loop over stages
            for (size_t i = 0; i < kernel_logn; i++, hsize *= 2, tt /= 2) {
                for (size_t j = 0, kstart = 0; j < hsize; j++, kstart += 2 * tt) {
                    // Compute twiddle factor exponent
                    uint32_t power = hsize + j;
                    uint32_t s;
                    
                    if (power == 0) {
                        s = 1;
                    } else if (power == (1 << (kernel_logn - 1))) {
                        s = kernel_root;
                    } else {
                        // Inline exponentiation: calculate s = root^power mod mod_val
                        uint32_t current_power = kernel_root;
                        uint32_t result = 1;
                        size_t shift_count = kernel_logn - 1;
                        
                        while (true) {
                            if (power & ((uint32_t)1 << shift_count)) {
                                // Equivalent to mul_mod(current_power, result, mod)
                                uint32_t product[2];
                                
                                // Equivalent to mul_uint_wide(current_power, result, product)
                                uint64_t res_temp = (uint64_t)current_power * (uint64_t)result;
                                product[0] = (uint32_t)(res_temp & 0xFFFFFFFF);
                                product[1] = (uint32_t)((res_temp >> 32) & 0xFFFFFFFF);
                                
                                // Equivalent to barrett_reduce_wide(product, mod)
                                // Which calls barrett_reduce_64input_32modulus(product, mod)
                                
                                // Round 1
                                uint32_t right_hw;
                                {
                                    uint32_t res[2];
                                    uint64_t res_temp = (uint64_t)product[0] * (uint64_t)kernel_const_ratio[0];
                                    res[0] = (uint32_t)(res_temp & 0xFFFFFFFF);
                                    res[1] = (uint32_t)((res_temp >> 32) & 0xFFFFFFFF);
                                    right_hw = res[1];
                                }

                                uint32_t middle_temp[2];
                                {
                                    uint64_t res_temp = (uint64_t)product[0] * (uint64_t)kernel_const_ratio[1];
                                    middle_temp[0] = (uint32_t)(res_temp & 0xFFFFFFFF);
                                    middle_temp[1] = (uint32_t)((res_temp >> 32) & 0xFFFFFFFF);
                                }
                                
                                uint32_t middle_lw;
                                uint32_t middle_lw_carry;
                                {
                                    middle_lw = right_hw + middle_temp[0];
                                    middle_lw_carry = (uint8_t)(middle_lw < right_hw);
                                }
                                
                                uint32_t middle_hw = middle_temp[1] + middle_lw_carry;

                                // Round 2
                                uint32_t middle2_temp[2];
                                {
                                    uint64_t res_temp = (uint64_t)product[1] * (uint64_t)kernel_const_ratio[0];
                                    middle2_temp[0] = (uint32_t)(res_temp & 0xFFFFFFFF);
                                    middle2_temp[1] = (uint32_t)((res_temp >> 32) & 0xFFFFFFFF);
                                }
                                
                                uint32_t middle2_lw;
                                uint32_t middle2_lw_carry;
                                {
                                    middle2_lw = middle_lw + middle2_temp[0];
                                    middle2_lw_carry = (uint8_t)(middle2_lw < middle_lw);
                                }
                                
                                uint32_t middle2_hw = middle2_temp[1] + middle2_lw_carry;

                                uint32_t tmp = product[1] * kernel_const_ratio[1] + middle_hw + middle2_hw;

                                // Barrett subtraction
                                tmp = product[0] - tmp * kernel_mod_val;
                                
                                // Equivalent to shift_result
                                int32_t is_2q = (int32_t)(tmp >= kernel_mod_val);
                                uint32_t mask = (uint32_t)(-is_2q);
                                result = (uint32_t)(tmp) - (kernel_mod_val & mask);
                            }
                            
                            power &= ~((uint32_t)1 << shift_count);
                            if (power == 0) {
                                s = result;
                                break;
                            }
                            
                            // Equivalent to mul_mod(current_power, current_power, mod)
                            uint32_t product[2];
                            
                            // Equivalent to mul_uint_wide(current_power, current_power, product)
                            uint64_t res_temp = (uint64_t)current_power * (uint64_t)current_power;
                            product[0] = (uint32_t)(res_temp & 0xFFFFFFFF);
                            product[1] = (uint32_t)((res_temp >> 32) & 0xFFFFFFFF);
                            
                            // Equivalent to barrett_reduce_wide(product, mod)
                            // Which calls barrett_reduce_64input_32modulus(product, mod)
                            
                            // Round 1
                            uint32_t right_hw;
                            {
                                uint32_t res[2];
                                uint64_t res_temp = (uint64_t)product[0] * (uint64_t)kernel_const_ratio[0];
                                res[0] = (uint32_t)(res_temp & 0xFFFFFFFF);
                                res[1] = (uint32_t)((res_temp >> 32) & 0xFFFFFFFF);
                                right_hw = res[1];
                            }

                            uint32_t middle_temp[2];
                            {
                                uint64_t res_temp = (uint64_t)product[0] * (uint64_t)kernel_const_ratio[1];
                                middle_temp[0] = (uint32_t)(res_temp & 0xFFFFFFFF);
                                middle_temp[1] = (uint32_t)((res_temp >> 32) & 0xFFFFFFFF);
                            }
                            
                            uint32_t middle_lw;
                            uint32_t middle_lw_carry;
                            {
                                middle_lw = right_hw + middle_temp[0];
                                middle_lw_carry = (uint8_t)(middle_lw < right_hw);
                            }
                            
                            uint32_t middle_hw = middle_temp[1] + middle_lw_carry;

                            // Round 2
                            uint32_t middle2_temp[2];
                            {
                                uint64_t res_temp = (uint64_t)product[1] * (uint64_t)kernel_const_ratio[0];
                                middle2_temp[0] = (uint32_t)(res_temp & 0xFFFFFFFF);
                                middle2_temp[1] = (uint32_t)((res_temp >> 32) & 0xFFFFFFFF);
                            }
                            
                            uint32_t middle2_lw;
                            uint32_t middle2_lw_carry;
                            {
                                middle2_lw = middle_lw + middle2_temp[0];
                                middle2_lw_carry = (uint8_t)(middle2_lw < middle_lw);
                            }
                            
                            uint32_t middle2_hw = middle2_temp[1] + middle2_lw_carry;

                            uint32_t tmp = product[1] * kernel_const_ratio[1] + middle_hw + middle2_hw;

                            // Barrett subtraction
                            tmp = product[0] - tmp * kernel_mod_val;
                            
                            // Equivalent to shift_result
                            int32_t is_2q = (int32_t)(tmp >= kernel_mod_val);
                            uint32_t mask = (uint32_t)(-is_2q);
                            current_power = (uint32_t)(tmp) - (kernel_mod_val & mask);
                            
                            shift_count--;
                        }
                    }
                    
                    // Process each pair in the current group
                    for (size_t k = kstart; k < (kstart + tt); k++) {
                        uint32_t u = data[k];
                        
                        // Equivalent to mul_mod(data[k + tt], s, mod)
                        uint32_t v;
                        {
                            uint32_t product[2];
                            
                            // Equivalent to mul_uint_wide(data[k + tt], s, product)
                            uint64_t res_temp = (uint64_t)data[k + tt] * (uint64_t)s;
                            product[0] = (uint32_t)(res_temp & 0xFFFFFFFF);
                            product[1] = (uint32_t)((res_temp >> 32) & 0xFFFFFFFF);
                            
                            // Equivalent to barrett_reduce_wide(product, mod)
                            // Which calls barrett_reduce_64input_32modulus(product, mod)
                            
                            // Round 1
                            uint32_t right_hw;
                            {
                                uint32_t res[2];
                                uint64_t res_temp = (uint64_t)product[0] * (uint64_t)kernel_const_ratio[0];
                                res[0] = (uint32_t)(res_temp & 0xFFFFFFFF);
                                res[1] = (uint32_t)((res_temp >> 32) & 0xFFFFFFFF);
                                right_hw = res[1];
                            }

                            uint32_t middle_temp[2];
                            {
                                uint64_t res_temp = (uint64_t)product[0] * (uint64_t)kernel_const_ratio[1];
                                middle_temp[0] = (uint32_t)(res_temp & 0xFFFFFFFF);
                                middle_temp[1] = (uint32_t)((res_temp >> 32) & 0xFFFFFFFF);
                            }
                            
                            uint32_t middle_lw;
                            uint32_t middle_lw_carry;
                            {
                                middle_lw = right_hw + middle_temp[0];
                                middle_lw_carry = (uint8_t)(middle_lw < right_hw);
                            }
                            
                            uint32_t middle_hw = middle_temp[1] + middle_lw_carry;

                            // Round 2
                            uint32_t middle2_temp[2];
                            {
                                uint64_t res_temp = (uint64_t)product[1] * (uint64_t)kernel_const_ratio[0];
                                middle2_temp[0] = (uint32_t)(res_temp & 0xFFFFFFFF);
                                middle2_temp[1] = (uint32_t)((res_temp >> 32) & 0xFFFFFFFF);
                            }
                            
                            uint32_t middle2_lw;
                            uint32_t middle2_lw_carry;
                            {
                                middle2_lw = middle_lw + middle2_temp[0];
                                middle2_lw_carry = (uint8_t)(middle2_lw < middle_lw);
                            }
                            
                            uint32_t middle2_hw = middle2_temp[1] + middle2_lw_carry;

                            uint32_t tmp = product[1] * kernel_const_ratio[1] + middle_hw + middle2_hw;

                            // Barrett subtraction
                            tmp = product[0] - tmp * kernel_mod_val;
                            
                            // Equivalent to shift_result
                            int32_t is_2q = (int32_t)(tmp >= kernel_mod_val);
                            uint32_t mask = (uint32_t)(-is_2q);
                            v = (uint32_t)(tmp) - (kernel_mod_val & mask);
                        }
                        
                        // Equivalent to add_mod(u, v, mod)
                        uint32_t result_add = u + v;
                        if (result_add >= kernel_mod_val) result_add -= kernel_mod_val;
                        data[k] = result_add;
                        
                        // Equivalent to sub_mod(u, v, mod)
                        // First, equivalent to neg_mod(v, mod)
                        uint32_t negated;
                        {
                            int32_t non_zero = (int32_t)(v != 0);
                            uint32_t mask = (uint32_t)(-non_zero);
                            negated = (kernel_mod_val - v) & mask;
                        }
                        
                        // Then, equivalent to add_mod(u, negated, mod)
                        uint32_t result_sub = u + negated;
                        if (result_sub >= kernel_mod_val) result_sub -= kernel_mod_val;
                        data[k + tt] = result_sub;
                    }
                }
            }
        });
    }
};

// Implementation of ntt 1 function
void ntt1(size_t n, size_t logn, uint32_t mod_value, const uint32_t* const_ratio, uint32_t *vec) {
    // Optionally, add input validation assertions as needed

    // Compute the primitive root for the NTT
    const uint32_t root = get_ntt_root(n, mod_value);

    // Create a SYCL buffer for the vector
    sycl::buffer<uint32_t, 1> vec_buf(vec, sycl::range<1>(n));

    // Choose the device selector (using the FPGA emulator selector)
    auto selector = sycl::ext::intel::fpga_emulator_selector_v;

    try {
        sycl::queue q{selector};
        std::cout << "Running NTT 1 on device: "
                  << q.get_device().get_info<sycl::info::device::name>().c_str()
                  << std::endl;

        // Submit work using a lambda that calls the NTTKernel functor
        q.submit([&](sycl::handler &h) {
            NTTKernel1(n, logn, mod_value, root, const_ratio, vec_buf)(h);
        }).wait();
    } catch (sycl::exception const &e) {
        std::cerr << "Caught a synchronous SYCL exception in ntt: "
                  << e.what() << "\n";
        std::exit(1);
    }
}

// NTT 2 Kernel functor class
class NTTKernel2 {
private:
    size_t n;
    size_t logn;
    uint32_t mod_value;
    uint32_t root;
    const uint32_t* const_ratio;
    mutable sycl::buffer<uint32_t, 1> vec_acc;

public:
    NTTKernel2(size_t n_val, size_t logn_val, uint32_t mod_val, uint32_t root_val, 
                const uint32_t* const_ratio_val, sycl::buffer<uint32_t, 1>& vec_buf)
        : n(n_val), logn(logn_val), mod_value(mod_val), root(root_val), 
            const_ratio(const_ratio_val), vec_acc(vec_buf) {}
    
    void operator()(sycl::handler& h) const {
        // Get access to the buffer
        auto data = vec_acc.get_access<sycl::access::mode::read_write>(h);
        
        // Capture necessary variables
        size_t kernel_n = n;
        size_t kernel_logn = logn;
        uint32_t kernel_mod_val = mod_value;
        uint32_t kernel_root = root;
        const uint32_t* kernel_const_ratio = const_ratio;
        
        h.single_task([=]() [[intel::kernel_args_restrict]] {
            size_t hsize = 1;
            size_t tt = kernel_n / 2;
            
            // Loop over stages
            for (size_t i = 0; i < kernel_logn; i++, hsize *= 2, tt /= 2) {
                for (size_t j = 0, kstart = 0; j < hsize; j++, kstart += 2 * tt) {
                    // Compute twiddle factor exponent
                    uint32_t power = hsize + j;
                    uint32_t s;
                    
                    if (power == 0) {
                        s = 1;
                    } else if (power == (1 << (kernel_logn - 1))) {
                        s = kernel_root;
                    } else {
                        // Inline exponentiation: calculate s = root^power mod mod_val
                        uint32_t current_power = kernel_root;
                        uint32_t result = 1;
                        size_t shift_count = kernel_logn - 1;
                        
                        while (true) {
                            if (power & ((uint32_t)1 << shift_count)) {
                                // Equivalent to mul_mod(current_power, result, mod)
                                uint32_t product[2];
                                
                                // Equivalent to mul_uint_wide(current_power, result, product)
                                uint64_t res_temp = (uint64_t)current_power * (uint64_t)result;
                                product[0] = (uint32_t)(res_temp & 0xFFFFFFFF);
                                product[1] = (uint32_t)((res_temp >> 32) & 0xFFFFFFFF);
                                
                                // Equivalent to barrett_reduce_wide(product, mod)
                                // Which calls barrett_reduce_64input_32modulus(product, mod)
                                
                                // Round 1
                                uint32_t right_hw;
                                {
                                    uint32_t res[2];
                                    uint64_t res_temp = (uint64_t)product[0] * (uint64_t)kernel_const_ratio[0];
                                    res[0] = (uint32_t)(res_temp & 0xFFFFFFFF);
                                    res[1] = (uint32_t)((res_temp >> 32) & 0xFFFFFFFF);
                                    right_hw = res[1];
                                }

                                uint32_t middle_temp[2];
                                {
                                    uint64_t res_temp = (uint64_t)product[0] * (uint64_t)kernel_const_ratio[1];
                                    middle_temp[0] = (uint32_t)(res_temp & 0xFFFFFFFF);
                                    middle_temp[1] = (uint32_t)((res_temp >> 32) & 0xFFFFFFFF);
                                }
                                
                                uint32_t middle_lw;
                                uint32_t middle_lw_carry;
                                {
                                    middle_lw = right_hw + middle_temp[0];
                                    middle_lw_carry = (uint8_t)(middle_lw < right_hw);
                                }
                                
                                uint32_t middle_hw = middle_temp[1] + middle_lw_carry;

                                // Round 2
                                uint32_t middle2_temp[2];
                                {
                                    uint64_t res_temp = (uint64_t)product[1] * (uint64_t)kernel_const_ratio[0];
                                    middle2_temp[0] = (uint32_t)(res_temp & 0xFFFFFFFF);
                                    middle2_temp[1] = (uint32_t)((res_temp >> 32) & 0xFFFFFFFF);
                                }
                                
                                uint32_t middle2_lw;
                                uint32_t middle2_lw_carry;
                                {
                                    middle2_lw = middle_lw + middle2_temp[0];
                                    middle2_lw_carry = (uint8_t)(middle2_lw < middle_lw);
                                }
                                
                                uint32_t middle2_hw = middle2_temp[1] + middle2_lw_carry;

                                uint32_t tmp = product[1] * kernel_const_ratio[1] + middle_hw + middle2_hw;

                                // Barrett subtraction
                                tmp = product[0] - tmp * kernel_mod_val;
                                
                                // Equivalent to shift_result
                                int32_t is_2q = (int32_t)(tmp >= kernel_mod_val);
                                uint32_t mask = (uint32_t)(-is_2q);
                                result = (uint32_t)(tmp) - (kernel_mod_val & mask);
                            }
                            
                            power &= ~((uint32_t)1 << shift_count);
                            if (power == 0) {
                                s = result;
                                break;
                            }
                            
                            // Equivalent to mul_mod(current_power, current_power, mod)
                            uint32_t product[2];
                            
                            // Equivalent to mul_uint_wide(current_power, current_power, product)
                            uint64_t res_temp = (uint64_t)current_power * (uint64_t)current_power;
                            product[0] = (uint32_t)(res_temp & 0xFFFFFFFF);
                            product[1] = (uint32_t)((res_temp >> 32) & 0xFFFFFFFF);
                            
                            // Equivalent to barrett_reduce_wide(product, mod)
                            // Which calls barrett_reduce_64input_32modulus(product, mod)
                            
                            // Round 1
                            uint32_t right_hw;
                            {
                                uint32_t res[2];
                                uint64_t res_temp = (uint64_t)product[0] * (uint64_t)kernel_const_ratio[0];
                                res[0] = (uint32_t)(res_temp & 0xFFFFFFFF);
                                res[1] = (uint32_t)((res_temp >> 32) & 0xFFFFFFFF);
                                right_hw = res[1];
                            }

                            uint32_t middle_temp[2];
                            {
                                uint64_t res_temp = (uint64_t)product[0] * (uint64_t)kernel_const_ratio[1];
                                middle_temp[0] = (uint32_t)(res_temp & 0xFFFFFFFF);
                                middle_temp[1] = (uint32_t)((res_temp >> 32) & 0xFFFFFFFF);
                            }
                            
                            uint32_t middle_lw;
                            uint32_t middle_lw_carry;
                            {
                                middle_lw = right_hw + middle_temp[0];
                                middle_lw_carry = (uint8_t)(middle_lw < right_hw);
                            }
                            
                            uint32_t middle_hw = middle_temp[1] + middle_lw_carry;

                            // Round 2
                            uint32_t middle2_temp[2];
                            {
                                uint64_t res_temp = (uint64_t)product[1] * (uint64_t)kernel_const_ratio[0];
                                middle2_temp[0] = (uint32_t)(res_temp & 0xFFFFFFFF);
                                middle2_temp[1] = (uint32_t)((res_temp >> 32) & 0xFFFFFFFF);
                            }
                            
                            uint32_t middle2_lw;
                            uint32_t middle2_lw_carry;
                            {
                                middle2_lw = middle_lw + middle2_temp[0];
                                middle2_lw_carry = (uint8_t)(middle2_lw < middle_lw);
                            }
                            
                            uint32_t middle2_hw = middle2_temp[1] + middle2_lw_carry;

                            uint32_t tmp = product[1] * kernel_const_ratio[1] + middle_hw + middle2_hw;

                            // Barrett subtraction
                            tmp = product[0] - tmp * kernel_mod_val;
                            
                            // Equivalent to shift_result
                            int32_t is_2q = (int32_t)(tmp >= kernel_mod_val);
                            uint32_t mask = (uint32_t)(-is_2q);
                            current_power = (uint32_t)(tmp) - (kernel_mod_val & mask);
                            
                            shift_count--;
                        }
                    }
                    
                    // Process each pair in the current group
                    for (size_t k = kstart; k < (kstart + tt); k++) {
                        uint32_t u = data[k];
                        
                        // Equivalent to mul_mod(data[k + tt], s, mod)
                        uint32_t v;
                        {
                            uint32_t product[2];
                            
                            // Equivalent to mul_uint_wide(data[k + tt], s, product)
                            uint64_t res_temp = (uint64_t)data[k + tt] * (uint64_t)s;
                            product[0] = (uint32_t)(res_temp & 0xFFFFFFFF);
                            product[1] = (uint32_t)((res_temp >> 32) & 0xFFFFFFFF);
                            
                            // Equivalent to barrett_reduce_wide(product, mod)
                            // Which calls barrett_reduce_64input_32modulus(product, mod)
                            
                            // Round 1
                            uint32_t right_hw;
                            {
                                uint32_t res[2];
                                uint64_t res_temp = (uint64_t)product[0] * (uint64_t)kernel_const_ratio[0];
                                res[0] = (uint32_t)(res_temp & 0xFFFFFFFF);
                                res[1] = (uint32_t)((res_temp >> 32) & 0xFFFFFFFF);
                                right_hw = res[1];
                            }

                            uint32_t middle_temp[2];
                            {
                                uint64_t res_temp = (uint64_t)product[0] * (uint64_t)kernel_const_ratio[1];
                                middle_temp[0] = (uint32_t)(res_temp & 0xFFFFFFFF);
                                middle_temp[1] = (uint32_t)((res_temp >> 32) & 0xFFFFFFFF);
                            }
                            
                            uint32_t middle_lw;
                            uint32_t middle_lw_carry;
                            {
                                middle_lw = right_hw + middle_temp[0];
                                middle_lw_carry = (uint8_t)(middle_lw < right_hw);
                            }
                            
                            uint32_t middle_hw = middle_temp[1] + middle_lw_carry;

                            // Round 2
                            uint32_t middle2_temp[2];
                            {
                                uint64_t res_temp = (uint64_t)product[1] * (uint64_t)kernel_const_ratio[0];
                                middle2_temp[0] = (uint32_t)(res_temp & 0xFFFFFFFF);
                                middle2_temp[1] = (uint32_t)((res_temp >> 32) & 0xFFFFFFFF);
                            }
                            
                            uint32_t middle2_lw;
                            uint32_t middle2_lw_carry;
                            {
                                middle2_lw = middle_lw + middle2_temp[0];
                                middle2_lw_carry = (uint8_t)(middle2_lw < middle_lw);
                            }
                            
                            uint32_t middle2_hw = middle2_temp[1] + middle2_lw_carry;

                            uint32_t tmp = product[1] * kernel_const_ratio[1] + middle_hw + middle2_hw;

                            // Barrett subtraction
                            tmp = product[0] - tmp * kernel_mod_val;
                            
                            // Equivalent to shift_result
                            int32_t is_2q = (int32_t)(tmp >= kernel_mod_val);
                            uint32_t mask = (uint32_t)(-is_2q);
                            v = (uint32_t)(tmp) - (kernel_mod_val & mask);
                        }
                        
                        // Equivalent to add_mod(u, v, mod)
                        uint32_t result_add = u + v;
                        if (result_add >= kernel_mod_val) result_add -= kernel_mod_val;
                        data[k] = result_add;
                        
                        // Equivalent to sub_mod(u, v, mod)
                        // First, equivalent to neg_mod(v, mod)
                        uint32_t negated;
                        {
                            int32_t non_zero = (int32_t)(v != 0);
                            uint32_t mask = (uint32_t)(-non_zero);
                            negated = (kernel_mod_val - v) & mask;
                        }
                        
                        // Then, equivalent to add_mod(u, negated, mod)
                        uint32_t result_sub = u + negated;
                        if (result_sub >= kernel_mod_val) result_sub -= kernel_mod_val;
                        data[k + tt] = result_sub;
                    }
                }
            }
        });
    }
};

// Implementation of ntt function
void ntt2(size_t n, size_t logn, uint32_t mod_value, const uint32_t* const_ratio, uint32_t *vec) {
    // Optionally, add input validation assertions as needed

    // Compute the primitive root for the NTT
    const uint32_t root = get_ntt_root(n, mod_value);

    // Create a SYCL buffer for the vector
    sycl::buffer<uint32_t, 1> vec_buf(vec, sycl::range<1>(n));

    // Choose the device selector (using the FPGA emulator selector)
    auto selector = sycl::ext::intel::fpga_emulator_selector_v;

    try {
        sycl::queue q{selector};
        std::cout << "Running NTT 2 on device: "
                  << q.get_device().get_info<sycl::info::device::name>().c_str()
                  << std::endl;

        // Submit work using a lambda that calls the NTTKernel functor
        q.submit([&](sycl::handler &h) {
            NTTKernel2(n, logn, mod_value, root, const_ratio, vec_buf)(h);
        }).wait();
    } catch (sycl::exception const &e) {
        std::cerr << "Caught a synchronous SYCL exception in ntt: "
                  << e.what() << "\n";
        std::exit(1);
    }
}

class PolyMultNTTKernel {
private:
    size_t n;
    uint32_t mod_value;
    const uint32_t* const_ratio;
    mutable sycl::buffer<uint32_t, 1> a_acc;
    mutable sycl::buffer<uint32_t, 1> b_acc;

public:
    PolyMultNTTKernel(size_t n_val, uint32_t mod_val, const uint32_t* const_ratio_val,
                        sycl::buffer<uint32_t, 1>& a_buf,
                        sycl::buffer<uint32_t, 1>& b_buf)
        : n(n_val), mod_value(mod_val), const_ratio(const_ratio_val), 
            a_acc(a_buf), b_acc(b_buf) {}
    
    void operator()(sycl::handler& h) const {
        // Get access to buffers
        auto a = a_acc.get_access<sycl::access::mode::read_write>(h);
        auto b = b_acc.get_access<sycl::access::mode::read>(h);
        
        // Capture necessary variables
        size_t kernel_n = n;
        uint32_t kernel_mod_val = mod_value;
        const uint32_t* kernel_const_ratio = const_ratio;
        
        h.single_task([=]() [[intel::kernel_args_restrict]] {
            // Component-wise multiplication
            for (size_t i = 0; i < kernel_n; i++) {
                // Get values
                uint32_t a_val = a[i];
                uint32_t b_val = b[i];
                
                // Multiply: a[i] = (a[i] * b[i]) mod q
                
                // 1. Multiply to get wide result
                uint32_t product[2];
                uint64_t res_temp = (uint64_t)a_val * (uint64_t)b_val;
                product[0] = (uint32_t)(res_temp & 0xFFFFFFFF);
                product[1] = (uint32_t)((res_temp >> 32) & 0xFFFFFFFF);
                
                // 2. Barrett reduction starts here
                
                // Round 1
                uint32_t right_hw;
                {
                    uint32_t res[2];
                    uint64_t res_temp = (uint64_t)product[0] * (uint64_t)kernel_const_ratio[0];
                    res[0] = (uint32_t)(res_temp & 0xFFFFFFFF);
                    res[1] = (uint32_t)((res_temp >> 32) & 0xFFFFFFFF);
                    right_hw = res[1];
                }

                uint32_t middle_temp[2];
                {
                    uint64_t res_temp = (uint64_t)product[0] * (uint64_t)kernel_const_ratio[1];
                    middle_temp[0] = (uint32_t)(res_temp & 0xFFFFFFFF);
                    middle_temp[1] = (uint32_t)((res_temp >> 32) & 0xFFFFFFFF);
                }
                
                uint32_t middle_lw;
                uint32_t middle_lw_carry;
                {
                    middle_lw = right_hw + middle_temp[0];
                    middle_lw_carry = (uint8_t)(middle_lw < right_hw);
                }
                
                uint32_t middle_hw = middle_temp[1] + middle_lw_carry;

                // Round 2
                uint32_t middle2_temp[2];
                {
                    uint64_t res_temp = (uint64_t)product[1] * (uint64_t)kernel_const_ratio[0];
                    middle2_temp[0] = (uint32_t)(res_temp & 0xFFFFFFFF);
                    middle2_temp[1] = (uint32_t)((res_temp >> 32) & 0xFFFFFFFF);
                }
                
                uint32_t middle2_lw;
                uint32_t middle2_lw_carry;
                {
                    middle2_lw = middle_lw + middle2_temp[0];
                    middle2_lw_carry = (uint8_t)(middle2_lw < middle_lw);
                }
                
                uint32_t middle2_hw = middle2_temp[1] + middle2_lw_carry;

                uint32_t tmp = product[1] * kernel_const_ratio[1] + middle_hw + middle2_hw;

                // Barrett subtraction
                tmp = product[0] - tmp * kernel_mod_val;
                
                // Final reduction if needed
                int32_t is_2q = (int32_t)(tmp >= kernel_mod_val);
                uint32_t mask = (uint32_t)(-is_2q);
                a[i] = (uint32_t)(tmp) - (kernel_mod_val & mask);
            }
        });
    }
};

// Public function to perform polynomial multiplication in NTT form
void ntt_form_poly_mod_mult(uint32_t *a, const uint32_t *b, size_t n, uint32_t mod_value, const uint32_t* const_ratio) {
    // Create SYCL buffers
    sycl::buffer<uint32_t, 1> a_buf(a, sycl::range<1>(n));
    sycl::buffer<uint32_t, 1> b_buf(const_cast<uint32_t*>(b), sycl::range<1>(n));
    
    // Create SYCL queue
    auto selector = sycl::ext::intel::fpga_emulator_selector_v;
    
    try {
        sycl::queue q{selector};
        
        // Print device info
        std::cout << "Running NTT Polynomial Multiplication on device: "
                  << q.get_device().get_info<sycl::info::device::name>().c_str()
                  << std::endl;
        
        // Submit and execute the kernel
        q.submit(PolyMultNTTKernel(n, mod_value, const_ratio, a_buf, b_buf)).wait();
        
    } catch (sycl::exception const &e) {
        std::cerr << "Caught a synchronous SYCL exception in poly_mult_mod_ntt_form_inpl: "
                  << e.what() << "\n";
        std::exit(1);
    }
}

class PolyNegModKernel {
private:
    size_t n;
    uint32_t mod_value;
    mutable sycl::buffer<uint32_t, 1> p_acc;

public:
    PolyNegModKernel(size_t n_val, uint32_t mod_val, sycl::buffer<uint32_t, 1>& p_buf)
        : n(n_val), mod_value(mod_val), p_acc(p_buf) {}
    
    void operator()(sycl::handler& h) const {
        // Get access to buffer
        auto p = p_acc.get_access<sycl::access::mode::read_write>(h);
        
        // Capture necessary variables
        size_t kernel_n = n;
        uint32_t kernel_mod_val = mod_value;
        
        h.single_task([=]() [[intel::kernel_args_restrict]] {
            // Apply negation to each coefficient
            for (size_t i = 0; i < kernel_n; i++) {
                // Get the coefficient
                uint32_t coeff = p[i];
                
                // Compute if coefficient is non-zero
                int32_t non_zero = (int32_t)(coeff != 0);
                uint32_t mask = (uint32_t)(-non_zero);
                
                // Compute negation: if coeff == 0, result = 0; else result = q - coeff
                uint32_t result = (kernel_mod_val - coeff) & mask;
                
                // Store the result
                p[i] = result;
            }
        });
    }
};

// Public function to negate polynomial coefficients modulo q
void poly_negate_mod(uint32_t *p, size_t n, uint32_t mod_value) {
    // Create SYCL buffer for the polynomial
    sycl::buffer<uint32_t, 1> p_buf(p, sycl::range<1>(n));
    
    // Create SYCL queue
    auto selector = sycl::ext::intel::fpga_emulator_selector_v;
    
    try {
        sycl::queue q{selector};
        
        // Print device info
        std::cout << "Running Polynomial Negation on device: "
                  << q.get_device().get_info<sycl::info::device::name>().c_str()
                  << std::endl;
        
        // Submit and execute the kernel
        q.submit(PolyNegModKernel(n, mod_value, p_buf)).wait();
        
    } catch (sycl::exception const &e) {
        std::cerr << "Caught a synchronous SYCL exception in poly_neg_mod: "
                  << e.what() << "\n";
        std::exit(1);
    }
}

// Kernel for modular addition of two polynomials
class PolyAddModKernel {
private:
    size_t n;
    uint32_t mod_value;
    mutable sycl::buffer<uint32_t, 1> p1_acc;
    mutable sycl::buffer<uint32_t, 1> p2_acc;

public:
    PolyAddModKernel(size_t n_val, uint32_t mod_val,
                        sycl::buffer<uint32_t, 1>& p1_buf,
                        sycl::buffer<uint32_t, 1>& p2_buf)
        : n(n_val), mod_value(mod_val), p1_acc(p1_buf), p2_acc(p2_buf) {}
    
    void operator()(sycl::handler& h) const {
        // Get access to buffers
        auto p1 = p1_acc.get_access<sycl::access::mode::read_write>(h);
        auto p2 = p2_acc.get_access<sycl::access::mode::read>(h);
        
        // Capture necessary variables
        size_t kernel_n = n;
        uint32_t kernel_mod_val = mod_value;
        
        h.single_task([=]() [[intel::kernel_args_restrict]] {
            // Process each coefficient
            for (size_t i = 0; i < kernel_n; i++) {
                // Get coefficients
                uint32_t coeff1 = p1[i];
                uint32_t coeff2 = p2[i];
                
                // Add coefficients
                uint32_t sum = coeff1 + coeff2;
                
                // Reduce modulo q: 
                // If sum >= q, subtract q
                int32_t is_ge_q = (int32_t)(sum >= kernel_mod_val);
                uint32_t mask = (uint32_t)(-is_ge_q);
                uint32_t result = sum - (kernel_mod_val & mask);
                
                // Store the result
                p1[i] = result;
            }
        });
    }
};
    
// Public function to add two polynomials modulo q
void poly_add_mod(uint32_t *p1, const uint32_t *p2, size_t n, uint32_t mod_value) {
    // Create SYCL buffers
    sycl::buffer<uint32_t, 1> p1_buf(p1, sycl::range<1>(n));
    sycl::buffer<uint32_t, 1> p2_buf(const_cast<uint32_t*>(p2), sycl::range<1>(n));
    
    // Create SYCL queue
    auto selector = sycl::ext::intel::fpga_emulator_selector_v;
    
    try {
        sycl::queue q{selector};
        
        // Print device info
        std::cout << "Running Polynomial Addition on device: "
                    << q.get_device().get_info<sycl::info::device::name>().c_str()
                    << std::endl;
        
        // Submit and execute the kernel
        q.submit(PolyAddModKernel(n, mod_value, p1_buf, p2_buf)).wait();
        
    } catch (sycl::exception const &e) {
        std::cerr << "Caught a synchronous SYCL exception in poly_add_mod_inpl: "
                    << e.what() << "\n";
        std::exit(1);
    }
}

// Function to run all kernels with proper data flow using pipe-based data transfer
void pipe_based_processing_pipeline(
    double scale,
    size_t n,
    size_t logn,
    uint32_t mod_value,
    const uint32_t* const_ratio,
    sycl::buffer<complex_double, 1>& encoding_buf,
    sycl::buffer<int8_t, 1>& error_samples_buf,
    sycl::buffer<uint32_t, 1>& ntt_pte_buf
) {
    // Create SYCL queue for FPGA
    sycl::queue q{sycl::ext::intel::fpga_emulator_selector_v};
    
    std::cout << "Running pipeline kernels in parallel on device: "
              << q.get_device().get_info<sycl::info::device::name>().c_str()
              << std::endl;
    
    try {
        // Submit all kernels concurrently
        std::cout << "Submitting all kernels concurrently..." << std::endl;
        
        // Create a vector to store events
        std::vector<sycl::event> events;
        
        // Submit all kernels WITHOUT waiting between submissions
        events.push_back(q.submit([&](sycl::handler& h) {
            std::cout << "Submit: IFFT kernel" << std::endl;
            IFFTKernel ifft_kernel(n, logn, encoding_buf, error_samples_buf);
            ifft_kernel(h);
        }));
        
        events.push_back(q.submit([&](sycl::handler& h) {
            std::cout << "Submit: Scale kernel" << std::endl;
            ScaleAndConvertKernel scale_kernel(n, scale);
            scale_kernel(h);
        }));
        
        events.push_back(q.submit([&](sycl::handler& h) {
            std::cout << "Submit: Reduction kernel" << std::endl;
            PTEReducePipeKernel pte_reduce(n, mod_value, const_ratio, ntt_pte_buf);
            pte_reduce(h);
        }));
        
        // Wait for all kernels to complete
        std::cout << "Waiting for all kernels to complete..." << std::endl;
        
        // Loop through each event and wait for completion
        for (size_t i = 0; i < events.size(); i++) {
            std::cout << "Waiting for kernel #" << (i+1) << std::endl;
            events[i].wait();
            std::cout << "Kernel #" << (i+1) << " completed" << std::endl;
        }
        
        std::cout << "All pipeline kernels completed successfully" << std::endl;
        
    } catch (sycl::exception const &e) {
        std::cerr << "SYCL exception caught in pipeline: " << e.what() << std::endl;
        throw; // Re-throw to caller
    }
}

// Implementation of the C-compatible function
extern "C" void SYCL_combined_encrypt(
    /* parms related values */
    size_t n,                           // Polynomial degree
    size_t logn,                        // Log of polynomial degree
    double scale,                       // Scale value
    uint32_t mod_value,                 // Modulus value (q)
    const uint32_t* const_ratio,        // Const ratio for Barrett reduction
    complex_double* encoding_buffer,    // Buffer for encoding
    uint32_t* expanded_s,               // Expanded secret key
    uint32_t* uniform_poly,             // Uniform polynomial (c1)
    int8_t* error_samples,              // Error samples
    int64_t* pt_with_error,             // Plaintext + error
    uint32_t* ntt_pte,                  // Scratch space for NTT
    uint32_t* c0_s,                     // Output: 1st ciphertext component
    uint32_t* c1,                       // Output: 2nd ciphertext component
    uint32_t* s_save,                   // Optional: Save expanded s (for testing)
    uint32_t* c1_save                   // Optional: Save c1 (for testing)
) {
    // Create SYCL buffers from the input pointers.
    sycl::buffer<complex_double, 1> encoding_buf(encoding_buffer, sycl::range<1>(n));
    sycl::buffer<int64_t, 1> pt_with_error_buf(pt_with_error, sycl::range<1>(n));
    sycl::buffer<uint32_t, 1> ntt_pte_buf(ntt_pte, sycl::range<1>(n));
    sycl::buffer<int8_t, 1> error_samples_buf(error_samples, sycl::range<1>(n));
    sycl::buffer<uint32_t, 1> expanded_s_buf(expanded_s, sycl::range<1>(n));
    sycl::buffer<uint32_t, 1> uniform_poly_buf(uniform_poly, sycl::range<1>(n));

    // Get host-access pointers for the expanded secret key and uniform poly.
    auto expanded_s_ptr = expanded_s_buf.get_host_access().get_pointer();
    auto uniform_poly_ptr = uniform_poly_buf.get_host_access().get_pointer();

    // Create a SYCL queue using the FPGA emulator selector.
    sycl::queue q{sycl::ext::intel::fpga_emulator_selector_v};
    auto device = q.get_device();

    // Pipeline the processing of encoding and error samples
    // This will run the IFFT and scaling in parallel with the reduction.
    // The encoding buffer is used for the IFFT, and the error samples are
    // used for the scaling and reduction.
    // The output of the IFFT is stored in the encoding buffer, and the
    // output of the scaling is stored in the pt_with_error buffer.
    // The reduction is done in place in the ntt_pte buffer.
    // The final result is stored in the ntt_pte buffer.
    pipe_based_processing_pipeline(scale, n, logn, mod_value, const_ratio, 
        encoding_buf, error_samples_buf, ntt_pte_buf);
    
    // ==============================================================
    //   Generate ciphertext components
    // ==============================================================
    
    // 1. Copy uniform polynomial to c1 output.
    std::memcpy(c1, uniform_poly_ptr, n * sizeof(uint32_t));
    
    // 2. Save c1 if requested for testing.
    if (c1_save != nullptr) {
        std::memcpy(c1_save, uniform_poly_ptr, n * sizeof(uint32_t));
    }
    
    // 3. Copy expanded secret key to c0_s.
    std::memcpy(c0_s, expanded_s_ptr, n * sizeof(uint32_t));
    
    // 4. Apply NTT to the secret key.
    // Updated call passing explicit parameters.
    ntt1(n, logn, mod_value, const_ratio, c0_s);
    
    // 5. Save NTT(s) for later decryption if requested.
    if (s_save != nullptr) {
        std::memcpy(s_save, c0_s, n * sizeof(uint32_t));
    }
    
    // 6. Calculate [a*s]_Rq using polynomial multiplication in NTT form.
    ntt_form_poly_mod_mult(c0_s, c1, n, mod_value, const_ratio);
    
    // 7. Negate [a*s]_Rq to get [-a*s]_Rq.
    poly_negate_mod(c0_s, n, mod_value);
    
    // 8. Process plaintext + error into ntt_pte.
    //reduce_pte(pt_with_error, n, mod_value, const_ratio, ntt_pte);
    
    // 9. Apply NTT to plaintext + error.
    ntt2(n, logn, mod_value, const_ratio, ntt_pte);
    
    // 10. Add to ciphertext.
    poly_add_mod(c0_s, ntt_pte, n, mod_value);
}

static uint32_t get_ntt_root(size_t n, uint32_t q)
{
    uint32_t root;
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
                    printf("Modulus value = %d", q);
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
                    printf("Modulus value = %d", q);
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
                    printf("Modulus value = %d", q);
                    exit(1);
                }
            }
            break;
        default: {
            printf("Error! Need first power of root for ntt\n");
            printf("Modulus value = %d", q);
            exit(1);
        }
    }
    return root;
}
