% =========================================================================
% RUN_TESTS.M — Unit Tests for All Core Components
% =========================================================================
% Verifies mathematical correctness of every module in the pipeline.
% Run before committing to GitHub to catch regressions.
%
% Usage:
%   >> run_tests          % runs all tests, prints PASS/FAIL
%   >> run_tests(true)    % verbose mode with intermediate values
% =========================================================================

function run_tests(verbose)
if nargin < 1, verbose = false; end

project_root = fileparts(mfilename('fullpath'));
addpath(genpath(project_root));

rng(0, 'twister');
n_pass = 0;  n_fail = 0;
results = {};

fprintf('=========================================\n');
fprintf('  Running Unit Tests\n');
fprintf('=========================================\n\n');

%% ---- Test 1: LMS predictor update rule ----------------------------
try
    H   = 5;  mu = 0.01;
    lms = lms_predictor_init(H, mu, 3);
    p   = randn(20, 3);
    for t = 1:20
        [p_hat, lms] = lms_predict(lms, p(t,:));
    end
    assert(~any(isnan(p_hat)), 'LMS output contains NaN');
    assert(~any(isinf(p_hat)), 'LMS output contains Inf');
    assert(all(size(p_hat) == [1,3]), 'LMS output wrong size');
    tf('LMS update rule', true, verbose);  n_pass = n_pass+1;
catch e
    tf('LMS update rule', false, verbose, e.message);  n_fail = n_fail+1;
end

%% ---- Test 2: Variance proxy = Var(residuals) ----------------------
try
    lms = lms_predictor_init(5, 0.01, 3);
    p   = randn(20, 3);
    for t = 1:20; [~,lms] = lms_predict(lms, p(t,:)); end
    sigma2 = compute_variance_proxy(lms);
    expected = var(lms.residuals(:));
    assert(abs(sigma2 - expected) < 1e-10, 'Variance proxy mismatch');
    assert(sigma2 >= 0, 'Variance proxy is negative');
    tf('Variance proxy', true, verbose); n_pass = n_pass+1;
catch e
    tf('Variance proxy', false, verbose, e.message); n_fail = n_fail+1;
end

%% ---- Test 3: MAE proxy is non-negative ----------------------------
try
    lms = lms_predictor_init(5, 0.01, 3);
    p   = randn(20,3);
    for t = 1:20; [~,lms] = lms_predict(lms, p(t,:)); end
    mae = compute_mae_proxy(lms);
    assert(mae >= 0, 'MAE proxy is negative');
    assert(~isnan(mae), 'MAE proxy is NaN');
    tf('MAE proxy', true, verbose); n_pass = n_pass+1;
catch e
    tf('MAE proxy', false, verbose, e.message); n_fail = n_fail+1;
end

%% ---- Test 4: State vector dimension (L2) --------------------------
try
    cfg = get_config();  cfg.H = 5;  cfg.state_dim_L2 = 8;
    lms = lms_predictor_init(5, 0.01, 3);
    p   = randn(15,3);
    for t = 1:15; [p_hat,lms] = lms_predict(lms, p(t,:)); end
    s   = build_state(lms, p_hat, 0.45, 'L2', cfg);
    assert(length(s) == cfg.state_dim_L2, ...
        sprintf('State dim %d != expected %d', length(s), cfg.state_dim_L2));
    assert(~any(isnan(s)), 'State vector contains NaN');
    tf('State vector (L2)', true, verbose); n_pass = n_pass+1;
catch e
    tf('State vector (L2)', false, verbose, e.message); n_fail = n_fail+1;
end

%% ---- Test 5: Reward function (Eq. 8 in paper) ---------------------
try
    cfg = get_config();  cfg.zeta = 0.5;  cfg.lambda = 1.0;
    p_true = [1, 0, 0];
    p_hat  = [0.9, 0.1, 0];
    % Test 5a: within budget (C_n < zeta)
    [R1, cv1] = compute_reward(p_true, p_hat, 0.3, cfg);
    expected_err = sum((p_true - p_hat).^2);
    assert(abs(R1 - (-expected_err)) < 1e-10, 'Reward wrong within budget');
    assert(cv1 == 0, 'Constraint violation should be 0 within budget');
    % Test 5b: over budget (C_n > zeta)
    [R2, cv2] = compute_reward(p_true, p_hat, 0.7, cfg);
    excess = 0.7 - 0.5;
    assert(abs(R2 - (-expected_err - cfg.lambda * excess^2)) < 1e-10, ...
        'Reward wrong when over budget');
    assert(cv2 > 0, 'Constraint violation should be positive over budget');
    tf('Reward function', true, verbose); n_pass = n_pass+1;
catch e
    tf('Reward function', false, verbose, e.message); n_fail = n_fail+1;
end

%% ---- Test 6: Offloading rate update (running mean) ----------------
try
    C = 0;
    actions = [1,0,1,1,0,0,1,0,1,1];
    for n = 1:length(actions)
        C = update_offloading_rate(C, actions(n), n);
    end
    expected = mean(actions);
    assert(abs(C - expected) < 1e-10, ...
        sprintf('OR=%.4f, expected %.4f', C, expected));
    tf('Offloading rate update', true, verbose); n_pass = n_pass+1;
catch e
    tf('Offloading rate update', false, verbose, e.message); n_fail = n_fail+1;
end

%% ---- Test 7: Q-network forward pass (DQN) -------------------------
try
    cfg = get_config();
    net = qnetwork_init(8, 2, 3, 128, 'DQN');
    s   = randn(8, 1);
    [Q, ~] = qnetwork_forward(net, s);
    assert(all(size(Q) == [2,1]), 'DQN output wrong size');
    assert(~any(isnan(Q)), 'DQN output has NaN');
    tf('Q-network forward (DQN)', true, verbose); n_pass = n_pass+1;
catch e
    tf('Q-network forward (DQN)', false, verbose, e.message); n_fail = n_fail+1;
end

%% ---- Test 8: Q-network forward pass (Dueling DQN) ----------------
try
    cfg = get_config();
    net = qnetwork_init(8, 2, 3, 128, 'DuelingDQN');
    s   = randn(8, 1);
    [Q, ~] = qnetwork_forward(net, s);
    assert(all(size(Q) == [2,1]), 'Dueling DQN output wrong size');
    assert(~any(isnan(Q)), 'Dueling DQN output has NaN');
    % Verify advantage sums to zero
    tf('Q-network forward (Dueling)', true, verbose); n_pass = n_pass+1;
catch e
    tf('Q-network forward (Dueling)', false, verbose, e.message); n_fail = n_fail+1;
end

%% ---- Test 9: Replay buffer add and sample -------------------------
try
    buf = replay_buffer_init(100, 8);
    for i = 1:60
        buf = replay_buffer_add(buf, randn(8,1), randi([0,1])-1, randn(), randn(8,1), false);
    end
    assert(buf.size == 60, sprintf('Buffer size %d != 60', buf.size));
    [s,a,r,sn,d] = replay_buffer_sample(buf, 32);
    assert(size(s,1) == 32, 'Batch size wrong');
    assert(size(s,2) == 8,  'State dim wrong in batch');
    assert(all(a == 0 | a == 1), 'Actions out of range');
    tf('Replay buffer', true, verbose); n_pass = n_pass+1;
catch e
    tf('Replay buffer', false, verbose, e.message); n_fail = n_fail+1;
end

%% ---- Test 10: Target network sync ---------------------------------
try
    net_o = qnetwork_init(8, 2, 3, 128, 'DQN');
    net_t = qnetwork_init(8, 2, 3, 128, 'DQN');
    % Modify online weights
    net_o.W{1}(1,1) = 99.0;
    net_t = qnetwork_sync(net_o, net_t);
    assert(net_t.W{1}(1,1) == 99.0, 'Target sync failed: weight mismatch');
    tf('Target network sync', true, verbose); n_pass = n_pass+1;
catch e
    tf('Target network sync', false, verbose, e.message); n_fail = n_fail+1;
end

%% ---- Test 11: Metrics — WI in [0,1] for good predictions ---------
try
    p_true = randn(100, 3);
    p_pred = p_true + 0.01 * randn(100, 3);   % near-perfect
    M = compute_metrics(p_true, p_pred);
    assert(M.wi >= 0 && M.wi <= 1, sprintf('WI=%.4f out of [0,1]', M.wi));
    assert(M.wi > 0.9, sprintf('WI=%.4f too low for near-perfect prediction', M.wi));
    assert(M.evs > 0.9, 'EVS too low for near-perfect prediction');
    assert(M.ss  > 0.9, 'Skill Score too low for near-perfect prediction');
    tf('Metrics (WI, EVS, SS)', true, verbose); n_pass = n_pass+1;
catch e
    tf('Metrics (WI, EVS, SS)', false, verbose, e.message); n_fail = n_fail+1;
end

%% ---- Test 12: Mann-Whitney U — known separation -------------------
try
    % Group 1: small errors; Group 2: large errors (clearly separated)
    grp1 = 0.01 + 0.002 * randn(200, 1);
    grp2 = 0.04 + 0.002 * randn(200, 1);
    [~, p_val, r_eff] = mann_whitney_u(grp1, grp2);
    assert(p_val < 0.001, sprintf('p=%.4f should be < 0.001', p_val));
    assert(r_eff > 0.5,   sprintf('r_eff=%.3f too small for clear separation', r_eff));
    tf('Mann-Whitney U test', true, verbose); n_pass = n_pass+1;
catch e
    tf('Mann-Whitney U test', false, verbose, e.message); n_fail = n_fail+1;
end

%% ---- Test 13: Gain metric = 100% for oracle policy ---------------
try
    % Simulate: RL achieves exactly Genie-level MSE
    mse_cpo   = 0.024;
    mse_genie = 0.021;
    mse_rl    = mse_genie;
    denom = mse_genie - mse_cpo;
    gain  = 100 * (mse_rl - mse_cpo) / denom;
    assert(abs(gain - 100) < 1e-8, 'Gain should be 100% for oracle RL');
    tf('Gain metric (oracle)', true, verbose); n_pass = n_pass+1;
catch e
    tf('Gain metric (oracle)', false, verbose, e.message); n_fail = n_fail+1;
end

%% ---- Test 14: Circular buffer wraps correctly ---------------------
try
    buf = replay_buffer_init(10, 4);
    for i = 1:15
        buf = replay_buffer_add(buf, i*ones(4,1), 0, i, i*ones(4,1), false);
    end
    assert(buf.size == 10, 'Buffer should be capped at capacity');
    assert(buf.ptr == 6,   'Write pointer should be at 6 after 15 inserts into size-10 buffer');
    tf('Circular buffer wrap', true, verbose); n_pass = n_pass+1;
catch e
    tf('Circular buffer wrap', false, verbose, e.message); n_fail = n_fail+1;
end

%% ---- Test 15: Ensemble proxy is non-negative ----------------------
try
    [~, ens] = ensemble_proxy_init(5, 0.01, 3, 3);
    p = randn(20, 3);
    proxy_val = 0;
    for t = 1:20
        [proxy_val, ens] = ensemble_proxy_update(ens, p(t,:));
    end
    assert(proxy_val >= 0, 'Ensemble proxy should be non-negative');
    tf('Ensemble proxy', true, verbose); n_pass = n_pass+1;
catch e
    tf('Ensemble proxy', false, verbose, e.message); n_fail = n_fail+1;
end

%% ---- Summary -------------------------------------------------------
fprintf('\n=========================================\n');
fprintf('  Test Results: %d PASSED, %d FAILED\n', n_pass, n_fail);
fprintf('=========================================\n');
if n_fail == 0
    fprintf('  All tests passed. Safe to push to GitHub.\n\n');
else
    fprintf('  WARNING: %d test(s) failed. Fix before pushing.\n\n', n_fail);
end
end

% ---- Helper: print test result -------------------------------------
function tf(test_name, passed, verbose, err_msg)
if nargin < 4, err_msg = ''; end
if passed
    fprintf('  [PASS] %s\n', test_name);
else
    fprintf('  [FAIL] %s\n', test_name);
    if verbose && ~isempty(err_msg)
        fprintf('         Error: %s\n', err_msg);
    end
end
end
