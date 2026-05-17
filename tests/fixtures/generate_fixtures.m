% tests/fixtures/generate_fixtures.m
% Run once in MATLAB to generate .npz reference fixtures for parity tests.
% Usage: matlab -batch "run('/home/chang/pykwave/tests/fixtures/generate_fixtures.m')"

addpath(genpath('/home/chang/k-Wave'));

%% Fixture 1: homogeneous lossless, point-source p0
fprintf('Generating fixture_lossless.npz...\n');
Nx = 64; dx = 1e-3; Ny = 64; dy = 1e-3; c0 = 1500; rho0 = 1000;
kgrid = kWaveGrid(Nx, dx, Ny, dy);
kgrid.makeTime(c0, 0.3);
medium.sound_speed = c0; medium.density = rho0;
p0 = zeros(Nx, Ny); p0(Nx/2, Ny/2) = 1000;
source.p0 = p0;
mask = zeros(Nx, Ny); mask(Nx/4, :) = 1;
sensor.mask = mask;
sensor_data = kspaceFirstOrder2D(kgrid, medium, source, sensor, 'PlotSim', false);
save_fixture('fixture_lossless.npz', sensor_data, kgrid.dt, kgrid.Nt);
fprintf('Saved fixture_lossless.npz\n');

%% Fixture 2: absorbing medium
fprintf('Generating fixture_absorbing.npz...\n');
medium2 = medium; medium2.alpha_coeff = 0.5; medium2.alpha_power = 1.5;
sensor_data2 = kspaceFirstOrder2D(kgrid, medium2, source, sensor, 'PlotSim', false);
save_fixture('fixture_absorbing.npz', sensor_data2, kgrid.dt, kgrid.Nt);
fprintf('Saved fixture_absorbing.npz\n');

fprintf('All fixtures generated successfully.\n');

function save_fixture(fname, data, dt, Nt)
    % Saves sensor_data matrix to a NumPy-compatible .npz file
    % data: sensor response matrix (N_sensors x Nt)
    % Converts to CSV for intermediate storage, then uses npz_write if available

    fixture_path = fileparts(mfilename('fullpath'));
    output_file = fullfile(fixture_path, strrep(fname, '.npz', '.mat'));

    % First save as .mat for debugging
    save(output_file, 'data', 'dt', 'Nt');

    % Write data as CSV (intermediate format)
    csv_file = fullfile(fixture_path, strrep(fname, '.npz', '.csv'));
    writematrix(data, csv_file);

    % Convert CSV to NPZ using Python if available
    % This requires scipy/numpy, which pykwave should have
    py_script = sprintf(['import numpy as np; ' ...
        'import os; ' ...
        'data = np.genfromtxt(''%s'', delimiter='',''); ' ...
        'np.savez(''%s'', sensor_data=data); ' ...
        'os.remove(''%s'')'], ...
        csv_file, fullfile(fixture_path, fname), csv_file);

    try
        pyenv;  % Initialize Python if needed
        pyexec(py_script);
    catch
        fprintf('Warning: Could not convert CSV to NPZ via Python.\n');
        fprintf('Manual conversion needed: python -c "import numpy as np; data = np.genfromtxt(''%s'', delimiter='',''); np.savez(''%s'', sensor_data=data)"\n', csv_file, fullfile(fixture_path, fname));
    end
end
