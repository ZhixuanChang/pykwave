# pykwave — JAX implementation of k-Wave pseudospectral acoustic simulation.
# Based on the k-Wave MATLAB Toolbox (http://www.k-wave.org) by B.E. Treeby & B.T. Cox.
# SPDX-License-Identifier: LGPL-3.0-or-later
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
