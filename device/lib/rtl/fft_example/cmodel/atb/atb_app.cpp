// ------------------------------------------------------------------------- 
// High Level Design Compiler for Altera(R) FPGAs Version 25.1 (Release Build #6a12354d2f)
// Software model created on 2025-07-08 08:39:19
// Generation mode: Bit Accurate
// ------------------------------------------------------------------------- 
#undef WRITE_STM_FILES
#define WRITE_STM_FILES
#include "fft_example_DUT_atb.h"
#include <iostream>

static bool s_has_error = false;

namespace csl
{
void error(const char* msg)   { s_has_error = true; std::cout << "Error: " << msg << "\n"; }
void warning(const char* msg) { std::cout << "Warning: " << msg << "\n"; }
void info(const char* msg)    { std::cout << "Info: " << msg << "\n"; }
}

int main(int argc, char** argv)
{

    bool success = false;
    const bool run_individual_atbs = (argc > 1) ? (strcmp("runIndividualAtbs", argv[1]) == 0) : false;
    if (!run_individual_atbs)
    {
        fft_example_DUTATB device_atb;
        if (device_atb.run())
        {
            success = device_atb.compare();
        }
    }

    if (run_individual_atbs)
    {
        success = true;
    }

    if (!s_has_error && success)
    {
        if (run_individual_atbs)
        {
            csl::info("[fft_example_DUT] Success! All individual software model ATB results matched the Simulink simulation results.");
        }
        else
        {
            csl::info("[fft_example_DUT] Success! Device-level software model ATB results matched the Simulink simulation results.");
        }
    }
    else if (s_has_error)
    {
        csl::info("[fft_example_DUT] An error occurred before the software model ATB could complete.");
    }
    else
    {
        if (run_individual_atbs)
        {
            csl::info("[fft_example_DUT] Failed! At least one individual ATB result did not match the Simulink simulation results.");
        }
        else
        {
            csl::info("[fft_example_DUT] Failed! Device-level software model ATB results did not match the Simulink simulation results.");
        }
        csl::info("NOTE: Mismatches may be expected if the software model simulation mode does not match the DSPBA mode.");
    }

    return success ? 0 : (s_has_error ? 1 : 2);
}
