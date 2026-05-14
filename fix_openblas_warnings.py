#!/usr/bin/env python3
"""
Script to run analysis with OpenBLAS warnings suppressed.

This script sets appropriate environment variables to prevent OpenBLAS warnings
and then runs the complete analysis.
"""

import os
import sys

def setup_environment():
    """Set environment variables to fix OpenBLAS warnings"""
    
    # Set OpenMP number of threads to 1 to avoid conflicts
    os.environ['OMP_NUM_THREADS'] = '1'
    
    # Set OpenBLAS to use single thread
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    
    # Set Intel MKL to use single thread (if available)
    os.environ['MKL_NUM_THREADS'] = '1'
    
    # Set VECLIB maximum threads (for macOS)
    os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
    
    # Set NUMEXPR threads
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    
    # Alternative: You can also try these for better performance
    # but they might still show warnings:
    # os.environ['OMP_NUM_THREADS'] = '4'  # Use 4 threads
    # os.environ['OPENBLAS_NUM_THREADS'] = '4'
    
    print("Environment variables set to prevent OpenBLAS warnings:")
    print(f"OMP_NUM_THREADS = {os.environ['OMP_NUM_THREADS']}")
    print(f"OPENBLAS_NUM_THREADS = {os.environ['OPENBLAS_NUM_THREADS']}")
    print(f"MKL_NUM_THREADS = {os.environ['MKL_NUM_THREADS']}")
    print()

if __name__ == "__main__":
    # Set environment variables before importing numpy/scipy
    setup_environment()
    
    # Import and run the analysis
    try:
        from run_complete_analysis import main
        print("Running complete analysis with OpenBLAS warnings suppressed...")
        main()
    except ImportError:
        print("Error: Could not import run_complete_analysis module")
        print("Make sure all analysis scripts are in the same directory")
        sys.exit(1)