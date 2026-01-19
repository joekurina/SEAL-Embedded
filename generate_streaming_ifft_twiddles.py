#!/usr/bin/env python3
"""
Generate precomputed twiddle factors for streaming negacyclic IFFT.

This generates twiddles organized for the radix-2^2 feedforward FFT architecture
as used in fft2d.hpp. The twiddles are organized by [stage][stream][index].

For a 4-parallel FFT:
- Complex rotation stages are 1, 3, 5, 7, 9, 11 (odd stages)
- Each complex stage has 3 streams (for 4-parallel: streams 0, 1, 2)
- Each stream processes N/4 indices

For negacyclic IFFT, we use 2N-th roots of unity instead of N-th roots.

Usage: python generate_streaming_ifft_twiddles.py <N>
Where N is one of: 4096, 8192, 16384

Outputs:
  streaming_twiddles_cos.txt - organized as stage, stream, index
  streaming_twiddles_sin.txt - organized as stage, stream, index
"""

import math
import sys


def generate_streaming_twiddles(n):
    """
    Generate twiddle factors organized for streaming FFT architecture.

    Based on the Twiddle function from fft2d.hpp:
    - For 4-parallel, we have 3 streams per complex stage
    - Complex stages are odd: 1, 3, 5, ...
    - The twiddle angle calculation follows the fft2d pattern but uses 2N for negacyclic
    """
    logn = int(math.log2(n))
    if 2**logn != n:
        raise ValueError(f"N must be a power of 2, got {n}")

    points = 4  # 4-parallel
    num_streams = 3  # For 4-parallel
    num_indices = n // points  # N/4 indices per stream

    # For negacyclic IFFT, use 2N instead of N
    # The base angle is -2*pi / (2*N) = -pi / N
    two_n = 2 * n

    # Complex stages are odd: 1, 3, 5, ..., logn-2
    # For logn=12: stages 1, 3, 5, 7, 9 (5 complex stages)
    complex_stages = [s for s in range(1, logn - 1) if s % 2 == 1]
    num_complex_stages = len(complex_stages)

    print(f"N = {n}, logN = {logn}")
    print(f"Points (parallelism) = {points}")
    print(f"Number of streams = {num_streams}")
    print(f"Number of indices per stream = {num_indices}")
    print(f"Complex stages: {complex_stages}")
    print(f"Number of complex stages = {num_complex_stages}")

    # Organize twiddles as [stage_idx][stream][index]
    # stage_idx is 0, 1, 2, ... mapping to actual stages 1, 3, 5, ...
    cos_twiddles = []
    sin_twiddles = []

    for stage_idx, stage in enumerate(complex_stages):
        stage_cos = []
        stage_sin = []

        for stream in range(num_streams):
            stream_cos = []
            stream_sin = []

            # Multiplier based on stream (from fft2d.hpp Twiddle function)
            if stream == 0:
                multiplier = 2
            elif stream == 1:
                multiplier = 1
            else:  # stream == 2
                multiplier = 3

            for index in range(num_indices):
                # Calculate position following fft2d.hpp formula
                # pos = (1 << (stage - 1)) * multiplier *
                #       ((index + (size / 8) * phase) & (size / 4 / (1 << (stage - 1)) - 1))
                # For 4-parallel with no phase (stream < 3), phase = 0

                mask = (n // 4) // (1 << (stage - 1)) - 1
                pos = (1 << (stage - 1)) * multiplier * (index & mask)
                pos = pos & (n - 1)  # Modulo N

                # For negacyclic IFFT: angle = -2*pi / (2*N) * pos = -pi/N * pos
                theta = -math.pi / n * pos

                stream_cos.append(math.cos(theta))
                stream_sin.append(math.sin(theta))

            stage_cos.append(stream_cos)
            stage_sin.append(stream_sin)

        cos_twiddles.append(stage_cos)
        sin_twiddles.append(stage_sin)

    return cos_twiddles, sin_twiddles, complex_stages


def format_double(val):
    """Format a double with full precision (17 significant digits)."""
    return f"{val:.17g}"


def write_twiddles_to_file(twiddles, filename, complex_stages):
    """Write twiddles organized as [stage][stream][index] to file."""
    with open(filename, "w") as f:
        for stage_idx, stage in enumerate(complex_stages):
            for stream in range(len(twiddles[stage_idx])):
                values = twiddles[stage_idx][stream]
                line = ",".join(format_double(v) for v in values)
                f.write(f"// Stage {stage}, Stream {stream}\n")
                f.write(line + "\n")


def generate_header(n, cos_twiddles, sin_twiddles, complex_stages):
    """Generate C++ header file with the twiddle tables."""
    logn = int(math.log2(n))
    points = 4
    num_streams = 3
    num_indices = n // points
    num_complex_stages = len(complex_stages)

    header = f"""#pragma once

// ============================================================================
// Precomputed twiddle factors for {n}-point streaming negacyclic IFFT
// ============================================================================
//
// Organized for radix-2^2 feedforward FFT architecture (4-parallel).
// Twiddles are indexed by [stage][stream][index].
//
// N = {n}, logN = {logn}
// Points (parallelism) = {points}
// Number of complex stages = {num_complex_stages}
// Complex stages: {complex_stages}
// Streams per stage = {num_streams}
// Indices per stream = {num_indices}
//
// For negacyclic IFFT, we use 2N-th roots of unity.
// Base angle: -pi / N
// ============================================================================

#include <cstddef>

namespace sycl_ckks {{

constexpr size_t STREAMING_IFFT_N = {n};
constexpr size_t STREAMING_IFFT_LOGN = {logn};
constexpr size_t STREAMING_IFFT_POINTS = {points};
constexpr size_t STREAMING_IFFT_NUM_STAGES = {num_complex_stages};
constexpr size_t STREAMING_IFFT_NUM_STREAMS = {num_streams};
constexpr size_t STREAMING_IFFT_NUM_INDICES = {num_indices};

// Twiddle factors: cos component
// Indexed as STREAMING_IFFT_COS[stage_idx][stream][index]
// stage_idx maps to actual stages: {dict(enumerate(complex_stages))}
constexpr double STREAMING_IFFT_COS[{num_complex_stages}][{num_streams}][{num_indices}] = {{
"""

    for stage_idx, stage in enumerate(complex_stages):
        header += f"    // Stage index {stage_idx} (actual stage {stage})\n"
        header += "    {\n"
        for stream in range(num_streams):
            header += f"        // Stream {stream}\n"
            header += "        {"
            values = cos_twiddles[stage_idx][stream]
            # Write in chunks for readability
            chunk_size = 8
            for i in range(0, len(values), chunk_size):
                chunk = values[i : i + chunk_size]
                if i > 0:
                    header += "         "
                header += ", ".join(format_double(v) for v in chunk)
                if i + chunk_size < len(values):
                    header += ",\n"
            header += "},\n"
        header += "    },\n"

    header += "};\n\n"

    header += f"""// Twiddle factors: sin component
// Indexed as STREAMING_IFFT_SIN[stage_idx][stream][index]
constexpr double STREAMING_IFFT_SIN[{num_complex_stages}][{num_streams}][{num_indices}] = {{
"""

    for stage_idx, stage in enumerate(complex_stages):
        header += f"    // Stage index {stage_idx} (actual stage {stage})\n"
        header += "    {\n"
        for stream in range(num_streams):
            header += f"        // Stream {stream}\n"
            header += "        {"
            values = sin_twiddles[stage_idx][stream]
            chunk_size = 8
            for i in range(0, len(values), chunk_size):
                chunk = values[i : i + chunk_size]
                if i > 0:
                    header += "         "
                header += ", ".join(format_double(v) for v in chunk)
                if i + chunk_size < len(values):
                    header += ",\n"
            header += "},\n"
        header += "    },\n"

    header += """};

// Map actual stage number to stage index
inline constexpr int streaming_ifft_stage_to_idx(int stage) {
    return (stage - 1) / 2;
}

}  // namespace sycl_ckks
"""

    return header


def main():
    if len(sys.argv) != 2:
        print("Usage: python generate_streaming_ifft_twiddles.py <N>")
        print("Where N is one of: 4096, 8192, 16384")
        sys.exit(1)

    try:
        n = int(sys.argv[1])
    except ValueError:
        print(f"Error: '{sys.argv[1]}' is not a valid integer")
        sys.exit(1)

    if n not in (4096, 8192, 16384):
        print(f"Error: N must be one of 4096, 8192, 16384, got {n}")
        sys.exit(1)

    print(f"Generating streaming IFFT twiddle factors for N={n}...")

    cos_twiddles, sin_twiddles, complex_stages = generate_streaming_twiddles(n)

    # Generate header file
    header = generate_header(n, cos_twiddles, sin_twiddles, complex_stages)

    header_filename = f"SYCL_ifft_{n // 1024}k_streaming_roots.h"
    with open(header_filename, "w") as f:
        f.write(header)

    print(f"Written header to {header_filename}")

    # Also write raw values for debugging
    write_twiddles_to_file(cos_twiddles, "streaming_twiddles_cos.txt", complex_stages)
    write_twiddles_to_file(sin_twiddles, "streaming_twiddles_sin.txt", complex_stages)
    print(
        "Written raw values to streaming_twiddles_cos.txt and streaming_twiddles_sin.txt"
    )

    print("Written raw values to streaming_twiddles_cos.txt and streaming_twiddles_sin.txt")

if __name__ == "__main__":
    main()
