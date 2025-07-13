#!/bin/sh
check_error() {
    result=$?
    if [ $result -ne 0 ]; then
        echo "Error: $1"
        exit $result
    fi
}

# Read more: https://www.intel.com/content/www/us/en/programmable/documentation/mwh1391807965224.html#owr1568230510943
fpga_crossgen -Xsv -Xsv the_fft.xml --cpp_model the_fft.cpp -o the_fft.o -w -I. -I../fft_example/cmodel -DFPGA_EMULATOR -DCSL_SYCL
check_error "fpga_crossgen command failed"
# The command below can be modified as needed if combining multiple object files into a single library.
fpga_libtool the_fft.o --create the_fft.a
check_error "fpga_libtool command failed"
