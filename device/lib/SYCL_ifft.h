// IFFT Kernel functor class with enhanced debugging
class IFFTKernel {
    private:
        size_t n;
        size_t logn;
    
    public:
        IFFTKernel(size_t n_val, size_t logn_val)
            : n(n_val), logn(logn_val) {}
        
        void operator()(sycl::handler& h) const {
            // Capture necessary variables
            size_t kernel_n = n;
            size_t kernel_logn = logn;
            
            // Create a stream for debug output
            sycl::stream out(2048, 512, h);
            
            h.single_task([=]() [[intel::kernel_args_restrict]] {
                out << "IFFT: Kernel started" << sycl::flush;
                
                // Local array to store input data - use fixed size as in original
                complex_double encoding[16384]; // Use max size that could be needed
                
                out << "IFFT: Starting to read " << kernel_n << " input values" << sycl::flush;
                
                // Read data from input pipe
                for (size_t i = 0; i < kernel_n; i++) {
                    // Print progress at specific points to avoid too many messages
                    if (i == 0) {
                        out << "IFFT: About to read first value" << sycl::flush;
                    }
                    
                    encoding[i] = EntranceToIFFTPipe::read();
                    
                    if (i == 0) {
                        out << "IFFT: Successfully read first value" << sycl::flush;
                    } else if (i % 1000 == 0 || i == kernel_n-1) {
                        out << "IFFT: Read " << (i+1) << "/" << kernel_n << " values" << sycl::flush;
                    }
                }
                
                out << "IFFT: All input values read, starting computation" << sycl::flush;
                
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
                    out << "IFFT: Computing round " << (round+1) << "/" << kernel_logn << sycl::flush;
                    
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
                
                out << "IFFT: Computation complete, starting to write results" << sycl::flush;
                
                // Write the results to the output pipe
                for (size_t i = 0; i < kernel_n; i++) {
                    if (i == 0) {
                        out << "IFFT: About to write first value" << sycl::flush;
                    }
                    
                    IFFTToScaleAndConvertPipe::write(encoding[i]);
                    
                    if (i == 0) {
                        out << "IFFT: Successfully wrote first value" << sycl::flush;
                    } else if (i % 1000 == 0 || i == kernel_n-1) {
                        out << "IFFT: Written " << (i+1) << "/" << kernel_n << " values" << sycl::flush;
                    }
                }
                
                out << "IFFT: All output values written" << sycl::flush;
            });
        }
    };