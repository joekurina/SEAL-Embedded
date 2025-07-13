@echo off
:: Read more: https://www.intel.com/content/www/us/en/programmable/documentation/mwh1391807965224.html#owr1568230510943
fpga_crossgen -Xsv -Xsv the_fft.xml --cpp_model the_fft.cpp -o the_fft.obj -w -I. -I../fft_example/cmodel -DFPGA_EMULATOR -DCSL_SYCL
if %ERRORLEVEL% NEQ 0 echo "Error: fpga_crossgen failed" && exit /B %ERRORLEVEL%
:: The command below can be modified as needed if combining multiple object files into a single library.
fpga_libtool the_fft.obj --create the_fft.lib
if %ERRORLEVEL% NEQ 0 echo "Error: fpga_libtool failed" && exit /B %ERRORLEVEL%
