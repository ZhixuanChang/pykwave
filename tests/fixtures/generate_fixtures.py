"""
MATLAB fixture generator wrapper.

Run this script from MATLAB via:
  matlab -batch "run('/home/chang/pykwave/tests/fixtures/generate_fixtures.m')"

This will produce the .npz reference files expected by test_parity.py:
  - fixture_lossless.npz
  - fixture_absorbing.npz

The actual fixture data generation is performed by generate_fixtures.m using
the k-Wave MATLAB library.
"""
