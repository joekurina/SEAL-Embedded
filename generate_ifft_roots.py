#!/usr/bin/env python3
"""
Generate precomputed twiddle factors for negacyclic IFFT.

Usage: python generate_ifft_roots.py <N>
Where N is one of: 4096, 8192, 16384

Outputs:
  real_roots.txt - comma-separated real parts
  imag_roots.txt - comma-separated imaginary parts
"""

import math
import sys


def bitrev(x, numbits):
    """Bit-reverse an integer x with numbits bits."""
    result = 0
    for _ in range(numbits):
        result = (result << 1) | (x & 1)
        x >>= 1
    return result


def generate_roots(n):
    """
    Generate twiddle factors for negacyclic IFFT of size N.

    For index k (1 <= k < N):
        br = bitrev(k, log2(N))
        angle = -2*pi / (2*N) * br = -pi/N * br
        twiddle[k] = (cos(angle), sin(angle))

    Index 0 is unused (placeholder with value 1.0 + 0.0j).
    """
    logn = int(math.log2(n))
    if 2**logn != n:
        raise ValueError(f"N must be a power of 2, got {n}")

    neg_two_pi_over_2n = -2.0 * math.pi / (n * 2)

    real_parts = []
    imag_parts = []

    for k in range(n):
        if k == 0:
            # Index 0 unused, placeholder
            real_parts.append(1.0)
            imag_parts.append(0.0)
        else:
            br = bitrev(k, logn)
            angle = neg_two_pi_over_2n * br
            real_parts.append(math.cos(angle))
            imag_parts.append(math.sin(angle))

    return real_parts, imag_parts


def format_double(val):
    """Format a double with full precision (17 significant digits)."""
    return f"{val:.17g}"


def main():
    if len(sys.argv) != 2:
        print("Usage: python generate_ifft_roots.py <N>")
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

    print(f"Generating {n} twiddle factors for negacyclic IFFT...")

    real_parts, imag_parts = generate_roots(n)

    with open("real_roots.txt", "w") as f:
        f.write(",".join(format_double(v) for v in real_parts))

    with open("imag_roots.txt", "w") as f:
        f.write(",".join(format_double(v) for v in imag_parts))

    print(f"Written {n} roots to real_roots.txt and imag_roots.txt")


if __name__ == "__main__":
    main()
