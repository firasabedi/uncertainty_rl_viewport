function cfg = get_config()
% GET_CONFIG  Returns the full hyperparameter configuration struct.
%
%   cfg = get_config()
%
%   All values match those reported in the paper (AJSE-D-26-01259).
%   Modify this file to run ablation studies or sensitivity analyses.
%
% -------------------------------------------------------------------------

%% ---- Dataset settings -----------------------------------------------
cfg.datasets       = {'david_mmsys', 'xu_pami', 'xu_cvpr'};
cfg.train_dataset  = 'david_mmsys';   % training dataset
cfg.train_split    = 0.80;            % 80/20 temporal split

%% ---- LMS predictor --------------------------------------------------
cfg.H              = 5;       % history window length (Section 4.2)
cfg.lms_mu         = 0.01;    % LMS step size
cfg.lms_n_axes     = 3;       % x, y, z head-position axes

%% ---- Variance proxy -------------------------------------------------
% sigma^2_n = Var of last H prediction residuals (Eq. 6 in paper)
cfg.proxy_type     = 'variance';   % 'variance' | 'mae' | 'none' | 'ensemble'

%% ---- RL agent -------------------------------------------------------
cfg.agent_types    = {'DQN', 'DuelingDQN'};
cfg.state_types    = {'Pos', 'L2'};   % raw positions vs L2-displacement states

% Q-network architecture
cfg.n_hidden_layers = 3;
cfg.hidden_units    = 128;
cfg.activation      = 'relu';

% State vector dimension for L2-based state:
%   H displacement magnitudes + 1 predicted displacement
%   + 1 variance proxy + 1 offloading rate  => H+3
cfg.state_dim_L2   = cfg.H + 3;    % = 8 for H=5
cfg.state_dim_Pos  = cfg.H * cfg.lms_n_axes + 1 + 1;  % raw 3D positions

cfg.n_actions      = 2;   % 0=local, 1=offload

%% ---- Constrained MDP ------------------------------------------------
cfg.zeta           = 0.50;   % target offloading budget (Eq. 2)
cfg.lambda         = 1.00;   % penalty weight (Eq. 8)
cfg.gamma          = 0.95;   % discount factor

%% ---- Training -------------------------------------------------------
cfg.n_episodes     = 300;
cfg.batch_size     = 64;
cfg.buffer_size    = 50000;   % replay buffer capacity
cfg.lr             = 0.001;   % Adam learning rate
cfg.target_sync    = 500;     % steps between target-network updates
cfg.eps_start      = 1.00;
cfg.eps_decay      = 0.995;   % per-episode decay
cfg.eps_min        = 0.01;
cfg.n_seeds        = 3;       % seeds for statistical averaging

%% ---- Noise experiment (Section 5.9) ---------------------------------
cfg.noise_std      = 0.15;    % Gaussian noise std added to LMS output

%% ---- Multi-step experiment (Section 5.8) ---------------------------
cfg.max_horizon    = 10;
cfg.reward_K       = [1, 3, 5];   % K-step lookahead for reward shaping

%% ---- Latency sensitivity (Section 5.10) ----------------------------
cfg.latency_deltas = [0, 0.5, 1.0, 2.0];   % decision intervals

%% ---- Sensitivity grids ----------------------------------------------
cfg.H_grid          = [3, 5, 10, 20];
cfg.gamma_grid      = 0.85:0.02:0.99;
cfg.zeta_grid       = 0.20:0.05:0.80;
cfg.lambda_grid     = [0.1, 0.5, 1.0, 1.5, 2.0, 3.0];

%% ---- Output paths ---------------------------------------------------
cfg.fig_dir        = fullfile('results', 'figures');
cfg.log_dir        = fullfile('results', 'logs');
cfg.data_dir       = fullfile('data', 'raw');

%% ---- Display -------------------------------------------------------
cfg.verbose        = true;
cfg.save_figures   = true;
cfg.fig_format     = 'pdf';   % 'pdf' | 'png' | 'eps'
end
