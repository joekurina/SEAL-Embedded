cmake_minimum_required (VERSION 3.13)
project (software_model)

set(CMAKE_CXX_STANDARD 14)
# set by user as a hint
set (GMP_INC_PATH "" CACHE PATH "GMP include path (hint)")
set (GMP_LIB_PATH "" CACHE PATH "GMP library path (hint)")
set (MPFR_INC_PATH "" CACHE PATH "MPFR include path (hint)")
set (MPFR_LIB_PATH "" CACHE PATH "MPFR library path (hint)")

include_directories(".")
add_compile_options("$<$<CONFIG:DEBUG>:-DCSL_DEBUG>")
if (MSVC)
    add_compile_options(/bigobj)
else()
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

option(USE_GMP "Include and link against the GMP library for models that require arbitrary precision" ON)
option(USE_MPFR "Include and link against the MPFR library for models that require arbitrary precision floating point" OFF)

set(software_model_common_INC
    "support/csl.h"
    "support/csl_def.h"
    "flt_fft_example_DUT_addSubFusedBlock_typeSFloatIEEE_52_11_4_correctRounding_38560000x0ao30cd06cj6ok0dpzc.h"
    "flt_fft_example_DUT_castBlock_typeSFloatIEEE_52_11_typeSFloatIEEE_52_11_castMode0000226123642i229742iyc5.h"
    "flt_fft_example_DUT_addBlock_typeSFloatIEEE_52_11_typeSFloatIEEE_52_11_typeSFloa0000j6of0cd16ok0cp06hj0u.h"
    "flt_fft_example_DUT_multBlock_typeSFloatIEEE_52_11_typeSFloatIEEE_52_11_typeSFlo0000of0cdj6of0cd16ol0qcz.h"
    "flt_fft_example_DUT_subBlock_typeSFloatIEEE_52_11_typeSFloatIEEE_52_11_typeSFloa00000cdj6of0cd16ok0ct30u.h"
    "fft_example_DUT.h"
)

set(software_model_atb_INC
    "support/csl_io.h"
    "support/csl_io_def.h"
    "atb/fft_example_DUT_atb.h"
)

set(software_model_atb_SRC
    "atb/atb_app.cpp"
)

add_executable(atb_app ${software_model_common_INC} ${software_model_atb_INC} ${software_model_atb_SRC})

if (MSVC)
    set_property(TARGET atb_app PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
endif()


if(USE_GMP)
    find_path(GMP_INC
        NAMES gmp.h
        HINTS ${GMP_INC_PATH} "$ENV{DSPBA_ROOTDIR}/Libraries/software" "$ENV{DSPBA_ROOTDIR}/backend/Libraries/software"
    )
    find_library(GMP_LIB
        NAMES dspba_mpir gmp mpir
        HINTS ${GMP_LIB_PATH} "$ENV{DSPBA_ROOTDIR}/Libraries/software/lib" "$ENV{DSPBA_ROOTDIR}/backend/Libraries/software/lib"
    )
    find_file(GMP_DLL
        NAMES dspba_mpir.dll libdspba_mpir.so gmp.dll mpir.dll libgmp.so
        PATHS ${GMP_LIB_PATH} "$ENV{DSPBA_ROOTDIR}/Libraries/software/lib" "$ENV{DSPBA_ROOTDIR}/backend/Libraries/software/lib"
        NO_DEFAULT_PATH
    )

    target_compile_definitions(atb_app PUBLIC CSL_USE_GMP)
    target_include_directories(atb_app PUBLIC ${GMP_INC})
    target_link_libraries(atb_app PUBLIC ${GMP_LIB})

    add_custom_command(TARGET atb_app POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${GMP_DLL}"
        $<TARGET_FILE_DIR:atb_app>)

endif()

if(USE_MPFR)
    find_path(MPFR_INC
        NAMES mpfr.h
        HINTS ${MPFR_INC_PATH} "$ENV{DSPBA_ROOTDIR}/Libraries/software" "$ENV{DSPBA_ROOTDIR}/backend/Libraries/software"
    )
    find_library(MPFR_LIB
        NAMES mpfr dspba_mpfr
        HINTS ${MPFR_LIB_PATH} "$ENV{DSPBA_ROOTDIR}/Libraries/software/lib" "$ENV{DSPBA_ROOTDIR}/backend/Libraries/software/lib"
    )
    find_file(MPFR_DLL
        NAMES mpfr.dll dspba_mpfr.dll libdspba_mpfr.so libmpfr.so
        PATHS ${MPFR_LIB_PATH} "$ENV{DSPBA_ROOTDIR}/Libraries/software/lib" "$ENV{DSPBA_ROOTDIR}/backend/Libraries/software/lib"
        NO_DEFAULT_PATH
    )

    target_compile_definitions(atb_app PUBLIC CSL_USE_MPFR)
    target_include_directories(atb_app PUBLIC ${MPFR_INC})
    target_link_libraries(atb_app PUBLIC ${MPFR_LIB})

    add_custom_command(TARGET atb_app POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${MPFR_DLL}"
        $<TARGET_FILE_DIR:atb_app>)

endif()

install(TARGETS atb_app DESTINATION bin)
