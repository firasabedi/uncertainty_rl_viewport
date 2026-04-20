function data = load_dataset(dataset_name, cfg)
% LOAD_DATASET  Load and preprocess a head-movement dataset.
%
%   data = load_dataset(dataset_name, cfg)
%
%   Inputs:
%     dataset_name  - string: 'david_mmsys' | 'xu_pami' | 'xu_cvpr'
%     cfg           - configuration struct from get_config()
%
%   Output (struct):
%     data.positions   - [N_tasks x 3] float, head positions (x,y,z)
%     data.n_tasks     - total number of decision epochs
%     data.train_idx   - indices for training split
%     data.test_idx    - indices for test split
%     data.name        - dataset label string
%
% -------------------------------------------------------------------------
% DATASET FORMAT EXPECTED (place CSV files in data/raw/):
%
%   david_mmsys  ->  data/raw/david_mmsys.csv
%   xu_pami      ->  data/raw/xu_pami.csv
%   xu_cvpr      ->  data/raw/xu_cvpr.csv
%
%   CSV columns: user_id, video_id, timestamp, x, y, z
%
% If the CSV file is not found, a synthetic dataset is generated for
% development/testing purposes only.
% -------------------------------------------------------------------------

raw_path = fullfile(cfg.data_dir, [dataset_name, '.csv']);

if exist(raw_path, 'file')
    fprintf('[Dataset] Loading %s from %s ...\n', dataset_name, raw_path);
    raw = readmatrix(raw_path);
    % Columns: user_id(1), video_id(2), timestamp(3), x(4), y(5), z(6)
    positions = raw(:, 4:6);
else
    warning('[Dataset] %s not found at %s. Generating synthetic data.', ...
        dataset_name, raw_path);
    positions = generate_synthetic_data(dataset_name);
end

N = size(positions, 1);

%% ---- Temporal train/test split (80/20 per video sequence) ----------
% For simplicity in this implementation we apply a global split.
% For per-sequence splitting, see the commented block below.
split_idx  = floor(N * cfg.train_split);
train_idx  = 1 : split_idx;
test_idx   = (split_idx + 1) : N;

%% ---- Pack output struct --------------------------------------------
data.positions  = positions;
data.n_tasks    = N;
data.train_idx  = train_idx;
data.test_idx   = test_idx;
data.name       = dataset_name;

fprintf('[Dataset] %s | Total: %d | Train: %d | Test: %d\n', ...
    dataset_name, N, length(train_idx), length(test_idx));
end

% =========================================================================
function positions = generate_synthetic_data(dataset_name)
% GENERATE_SYNTHETIC_DATA
%   Produces head-movement traces with dataset-appropriate scale and
%   complexity for development and CI testing.  NOT for publication use.
% =========================================================================
rng(0);
switch lower(dataset_name)
    case 'david_mmsys'
        N = 19100;   % ~12.6k train + 6.5k test
        sigma_motion = 0.05;
    case 'xu_pami'
        N = 65000;   % >52k test samples
        sigma_motion = 0.08;
    case 'xu_cvpr'
        N = 340000;  % >270k test samples
        sigma_motion = 0.10;
    otherwise
        N = 20000;
        sigma_motion = 0.06;
end

% AR(1) model: P_t = 0.95*P_{t-1} + noise (smooth head motion)
alpha   = 0.95;
positions = zeros(N, 3);
positions(1,:) = randn(1,3) * 0.1;
for t = 2:N
    positions(t,:) = alpha * positions(t-1,:) + ...
                     sigma_motion * randn(1,3);
end
% Occasional abrupt saccades (non-stationary bursts)
saccade_times = randperm(N, floor(N * 0.03));
for t = saccade_times
    positions(t,:) = positions(t,:) + randn(1,3) * 3 * sigma_motion;
end
end
