function run_experiment(choice, cfg)
% RUN_EXPERIMENT  Dispatcher for all paper experiments.
%
%   run_experiment(choice, cfg)
%
%   choice = 1..9  runs a single experiment
%   choice = 10    runs all experiments sequentially
% -------------------------------------------------------------------------

if choice == 10
    for k = 1:9
        run_experiment(k, cfg);
    end
    return
end

switch choice
    case 1,  exp_training_convergence(cfg);
    case 2,  exp_ablation_proxy(cfg);
    case 3,  exp_sensitivity_H(cfg);
    case 4,  exp_sensitivity_hyperparams(cfg);
    case 5,  exp_indistribution(cfg);
    case 6,  exp_cross_dataset(cfg);
    case 7,  exp_multistep(cfg);
    case 8,  exp_robustness_noise(cfg);
    case 9,  exp_latency_sensitivity(cfg);
    otherwise
        error('Invalid experiment choice. Select 1-10.');
end
end


% =========================================================================
% EXPERIMENT 1: Training convergence (Figure 3 and 4 in paper)
% =========================================================================
function exp_training_convergence(cfg)

fprintf('\n[EXP 1] Training Convergence\n');
data   = load_dataset(cfg.train_dataset, cfg);
server = server_predictor_init(data.positions(data.train_idx,:), cfg);
pos_tr = data.positions(data.train_idx, :);

configs  = {'DQN','L2'; 'DQN','Pos'; 'DuelingDQN','L2'; 'DuelingDQN','Pos'};
labels   = {'DQN-L2','DQN-Pos','Dueling DQN-L2','Dueling DQN-Pos'};
colors   = {'b','c','r','m'};

for noise_mode = 0:1
    fig = figure('Visible','off');
    hold on; grid on;
    leg_handles = gobjects(size(configs,1),1);

    for k = 1:size(configs,1)
        atype = configs{k,1};
        stype = configs{k,2};

        % Average over n_seeds
        all_rewards = zeros(cfg.n_seeds, cfg.n_episodes);
        for s = 1:cfg.n_seeds
            cfg_s = cfg;
            if noise_mode == 1
                cfg_s.noise_std = cfg.noise_std;
            else
                cfg_s.noise_std = 0;
            end
            [~, ep_r, ~] = train_agent(pos_tr, server, cfg_s, atype, stype, s);
            all_rewards(s,:) = ep_r';
        end
        mean_r = mean(all_rewards, 1);
        h = plot(1:cfg.n_episodes, mean_r, colors{k}, 'LineWidth', 1.5);
        leg_handles(k) = h;
    end

    legend(leg_handles, labels, 'Location','northwest', 'FontSize', 9);
    xlabel('Training Episodes'); ylabel('Average Total Reward');
    cond_str = {'Clean','Noisy'};
    title(sprintf('Training Convergence under %s Conditions', cond_str{noise_mode+1}));
    save_figure(fig, sprintf('Fig%d_training_%s', 3+noise_mode, lower(cond_str{noise_mode+1})), cfg);
end
fprintf('[EXP 1] Done.\n');
end


% =========================================================================
% EXPERIMENT 2: Ablation study — reliability proxy variants (Table 4)
% =========================================================================
function exp_ablation_proxy(cfg)

fprintf('\n[EXP 2] Ablation: Reliability Proxy\n');
data   = load_dataset(cfg.train_dataset, cfg);
server = server_predictor_init(data.positions(data.train_idx,:), cfg);
pos_tr = data.positions(data.train_idx, :);
pos_te = data.positions(data.test_idx,  :);
srv_te = server_predictor_init(pos_te, cfg);

proxy_variants = {'none','mae','variance','ensemble'};
proxy_labels   = {'No Reliability Signal','MAE Proxy', ...
                  'Variance Proxy (Proposed)','Ensemble Disagreement'};

fprintf('%-32s %10s %10s %12s\n', 'Method','MSE','Gain(%)','Offload Rate');
fprintf('%s\n', repmat('-', 65, 1));

for pv = 1:length(proxy_variants)
    cfg_v             = cfg;
    cfg_v.proxy_type  = proxy_variants{pv};
    mse_v = zeros(cfg.n_seeds,1);
    or_v  = zeros(cfg.n_seeds,1);

    for s = 1:cfg.n_seeds
        [agent, ~, ~] = train_agent(pos_tr, server, cfg_v, 'DuelingDQN', 'L2', s);
        res = evaluate_agent(agent, pos_te, srv_te, cfg_v);
        mse_v(s) = res.mse_rl;
        or_v(s)  = res.offload_rate;
    end

    denom = mean([res.mse_genie]) - mean([res.mse_cpo]);
    gain  = 100*(mean(mse_v) - mean([res.mse_cpo])) / denom;

    fprintf('%-32s %10.5f %10.1f %12.3f\n', ...
        proxy_labels{pv}, mean(mse_v), gain, mean(or_v));
end
fprintf('[EXP 2] Done.\n');
end


% =========================================================================
% EXPERIMENT 3: Sensitivity to history length H (Table 5)
% =========================================================================
function exp_sensitivity_H(cfg)

fprintf('\n[EXP 3] Sensitivity to History Length H\n');
data   = load_dataset(cfg.train_dataset, cfg);

fprintf('%-5s %22s %10s %14s %15s\n', 'H','MSE (mean ± std)','Gain(%)','OR','FLOPs/ep');
fprintf('%s\n', repmat('-', 70, 1));

for H = cfg.H_grid
    cfg_h         = cfg;
    cfg_h.H       = H;
    cfg_h.state_dim_L2 = H + 3;
    mse_seeds = zeros(cfg.n_seeds, 1);
    or_seeds  = zeros(cfg.n_seeds, 1);

    pos_tr = data.positions(data.train_idx,:);
    pos_te = data.positions(data.test_idx, :);
    srv_tr = server_predictor_init(pos_tr, cfg_h);
    srv_te = server_predictor_init(pos_te, cfg_h);

    for s = 1:cfg.n_seeds
        [agent, ~, ~] = train_agent(pos_tr, srv_tr, cfg_h, 'DuelingDQN', 'L2', s);
        res           = evaluate_agent(agent, pos_te, srv_te, cfg_h);
        mse_seeds(s)  = res.mse_rl;
        or_seeds(s)   = res.offload_rate;
    end

    m_mse = mean(mse_seeds); s_mse = std(mse_seeds);
    ci_lo = m_mse - 1.96*s_mse/sqrt(cfg.n_seeds);
    ci_hi = m_mse + 1.96*s_mse/sqrt(cfg.n_seeds);
    denom = res.mse_genie - res.mse_cpo;
    gain  = 100*(m_mse - res.mse_cpo)/denom;
    flops = 2*cfg.n_hidden_layers*cfg.hidden_units^2 + 3*H + H + 3;

    fprintf('%-5d %8.5f (%6.5f--%6.5f) %10.1f %14.3f %15d\n', ...
        H, m_mse, ci_lo, ci_hi, gain, mean(or_seeds), flops);
end
fprintf('[EXP 3] Done.\n');
end


% =========================================================================
% EXPERIMENT 4: Sensitivity to gamma, zeta, lambda (Figure 5)
% =========================================================================
function exp_sensitivity_hyperparams(cfg)

fprintf('\n[EXP 4] Hyperparameter Sensitivity\n');
data   = load_dataset(cfg.train_dataset, cfg);
pos_tr = data.positions(data.train_idx,:);
pos_te = data.positions(data.test_idx, :);
srv_tr = server_predictor_init(pos_tr, cfg);
srv_te = server_predictor_init(pos_te, cfg);

param_names  = {'gamma','zeta','lambda'};
param_grids  = {cfg.gamma_grid, cfg.zeta_grid, cfg.lambda_grid};
param_labels = {'\gamma (discount)', '\zeta (budget)', '\lambda (penalty)'};
fig_tags     = {'gamma','zeta','lambda'};

fig = figure('Visible','off','Position',[0 0 1200 350]);
for pp = 1:3
    subplot(1,3,pp); hold on; grid on;
    grid_vals = param_grids{pp};
    mse_grid  = zeros(length(grid_vals), cfg.n_seeds);

    for gi = 1:length(grid_vals)
        cfg_p = cfg;
        cfg_p.(param_names{pp}) = grid_vals(gi);
        for s = 1:cfg.n_seeds
            [agent,~,~] = train_agent(pos_tr, srv_tr, cfg_p, 'DuelingDQN','L2',s);
            res         = evaluate_agent(agent, pos_te, srv_te, cfg_p);
            mse_grid(gi,s) = res.mse_rl;
        end
    end

    mu_g  = mean(mse_grid, 2);
    sd_g  = std(mse_grid, 0, 2);
    ci_g  = 1.96 * sd_g / sqrt(cfg.n_seeds);

    fill([grid_vals, fliplr(grid_vals)], ...
         [mu_g'+ci_g', fliplr(mu_g'-ci_g')], ...
         [0.8 0.9 1.0], 'EdgeColor','none', 'FaceAlpha',0.5);
    plot(grid_vals, mu_g, 'b-o', 'LineWidth',1.5, 'MarkerSize',5);
    xline(cfg.(param_names{pp}), 'r--', 'LineWidth',1.5);
    xlabel(param_labels{pp}); ylabel('MSE');
    title(sprintf('Performance vs. %s', param_labels{pp}));
end
save_figure(fig, 'Fig5_sensitivity_hyperparams', cfg);
fprintf('[EXP 4] Done.\n');
end


% =========================================================================
% EXPERIMENT 5: In-distribution results — David_MMSys (Table 6)
% =========================================================================
function exp_indistribution(cfg)

fprintf('\n[EXP 5] In-Distribution Results: %s\n', cfg.train_dataset);
data   = load_dataset(cfg.train_dataset, cfg);
pos_tr = data.positions(data.train_idx,:);
pos_te = data.positions(data.test_idx, :);
srv_tr = server_predictor_init(pos_tr, cfg);
srv_te = server_predictor_init(pos_te, cfg);

agent_cfgs = {'DQN','Pos'; 'DQN','L2'; 'DuelingDQN','Pos'; 'DuelingDQN','L2'};
labels     = {'DQN Pos','DQN L2','Dueling DQN Pos','Dueling DQN L2'};
all_res    = cell(size(agent_cfgs,1), 1);

for k = 1:size(agent_cfgs,1)
    atype = agent_cfgs{k,1}; stype = agent_cfgs{k,2};
    mse_s = zeros(cfg.n_seeds,1); or_s = zeros(cfg.n_seeds,1);
    last_res = [];
    for s = 1:cfg.n_seeds
        [agent,~,~] = train_agent(pos_tr, srv_tr, cfg, atype, stype, s);
        res = evaluate_agent(agent, pos_te, srv_te, cfg);
        mse_s(s) = res.mse_rl;  or_s(s) = res.offload_rate;
        last_res = res;
    end
    ci = 1.96 * std(mse_s) / sqrt(cfg.n_seeds);
    all_res{k} = struct('mse_mean', mean(mse_s), 'ci_lo', mean(mse_s)-ci, ...
        'ci_hi', mean(mse_s)+ci, 'or_mean', mean(or_s), ...
        'mse_sml', last_res.mse_sml, 'mse_lml', last_res.mse_lml, ...
        'mse_cpo', last_res.mse_cpo, 'mse_genie', last_res.mse_genie, ...
        'gain_mean', 100*(mean(mse_s)-last_res.mse_cpo)/(last_res.mse_genie-last_res.mse_cpo));
    fprintf('[%s-%s] MSE=%.5f ± %.5f | Gain=%.1f%% | OR=%.3f\n', ...
        atype, stype, mean(mse_s), ci, all_res{k}.gain_mean, mean(or_s));
end
print_results_table(all_res, labels, cfg.train_dataset);
fprintf('[EXP 5] Done.\n');
end


% =========================================================================
% EXPERIMENT 6: Cross-dataset generalization (Tables 8 and 9)
% =========================================================================
function exp_cross_dataset(cfg)

fprintf('\n[EXP 6] Cross-Dataset Generalization\n');
data_tr = load_dataset(cfg.train_dataset, cfg);
pos_tr  = data_tr.positions(data_tr.train_idx, :);
srv_tr  = server_predictor_init(pos_tr, cfg);

target_datasets = {'xu_pami', 'xu_cvpr'};

for dd = 1:length(target_datasets)
    dname   = target_datasets{dd};
    data_te = load_dataset(dname, cfg);
    pos_te  = data_te.positions;
    srv_te  = server_predictor_init(pos_te, cfg);

    fprintf('\n--- Cross-dataset: Train=%s | Test=%s ---\n', cfg.train_dataset, dname);
    agent_cfgs = {'DQN','Pos'; 'DQN','L2'; 'DuelingDQN','Pos'; 'DuelingDQN','L2'};
    for k = 1:size(agent_cfgs,1)
        atype = agent_cfgs{k,1}; stype = agent_cfgs{k,2};
        mse_s = zeros(cfg.n_seeds,1);
        for s = 1:cfg.n_seeds
            [agent,~,~] = train_agent(pos_tr, srv_tr, cfg, atype, stype, s);
            res = evaluate_agent(agent, pos_te, srv_te, cfg);
            mse_s(s) = res.mse_rl;
        end
        ci   = 1.96*std(mse_s)/sqrt(cfg.n_seeds);
        gain = 100*(mean(mse_s)-res.mse_cpo)/(res.mse_genie-res.mse_cpo);
        fprintf('[%s-%s] MSE=%.5f (%.5f--%.5f) | Gain=%.1f%%\n', ...
            atype, stype, mean(mse_s), mean(mse_s)-ci, mean(mse_s)+ci, gain);
    end
end
fprintf('[EXP 6] Done.\n');
end


% =========================================================================
% EXPERIMENT 7: Multi-step prediction + reward shaping (Section 5.8)
% =========================================================================
function exp_multistep(cfg)

fprintf('\n[EXP 7] Multi-Step Prediction\n');
data   = load_dataset(cfg.train_dataset, cfg);
pos_tr = data.positions(data.train_idx,:);
pos_te = data.positions(data.test_idx, :);
srv_tr = server_predictor_init(pos_tr, cfg);
srv_te = server_predictor_init(pos_te, cfg);

[agent,~,~] = train_agent(pos_tr, srv_tr, cfg, 'DuelingDQN', 'L2', 1);

horizons = 1:cfg.max_horizon;
mse_h    = zeros(length(horizons), 5);   % RL,CPO,SML,LML,Genie

for h = horizons
    % Approximate h-step MSE by rolling prediction
    mse_h(h,:) = evaluate_multistep(agent, pos_te, srv_te, cfg, h);
end

fig = figure('Visible','off'); hold on; grid on;
methods = {'RL','CPO','S-ML','L-ML','Genie'};
styles  = {'b-o','r-s','g-^','m-d','k-*'};
for m = 1:5
    plot(horizons, mse_h(:,m), styles{m}, 'LineWidth',1.5, 'MarkerSize',6);
end
legend(methods,'Location','northwest'); xlabel('Prediction Horizon (steps)');
ylabel('MSE'); title('Multi-Step Prediction Performance');
save_figure(fig, 'Fig7_multistep', cfg);

pct_increase = 100*(mse_h(5,1)-mse_h(1,1))/mse_h(1,1);
fprintf('[EXP 7] MSE 1->5 step increase: %.1f%%\n', pct_increase);
fprintf('[EXP 7] Done.\n');
end


% =========================================================================
% EXPERIMENT 8: Robustness to degraded local predictor (Table 10)
% =========================================================================
function exp_robustness_noise(cfg)

fprintf('\n[EXP 8] Robustness under Noisy Local Predictor\n');
cfg_n           = cfg;
cfg_n.noise_std = cfg.noise_std;

data   = load_dataset(cfg.train_dataset, cfg_n);
pos_tr = data.positions(data.train_idx,:);
pos_te = data.positions(data.test_idx, :);
srv_tr = server_predictor_init(pos_tr, cfg_n);
srv_te = server_predictor_init(pos_te, cfg_n);

agent_cfgs = {'DQN','Pos'; 'DQN','L2'; 'DuelingDQN','Pos'; 'DuelingDQN','L2'};
fprintf('%-28s %10s %10s %10s\n', 'Method','MSE','Gain(%)','OR');
for k = 1:size(agent_cfgs,1)
    atype = agent_cfgs{k,1}; stype = agent_cfgs{k,2};
    mse_s = zeros(cfg.n_seeds,1);
    for s = 1:cfg.n_seeds
        [agent,~,~] = train_agent(pos_tr, srv_tr, cfg_n, atype, stype, s);
        res = evaluate_agent(agent, pos_te, srv_te, cfg_n);
        mse_s(s) = res.mse_rl;
    end
    gain = 100*(mean(mse_s)-res.mse_cpo)/(res.mse_genie-res.mse_cpo);
    fprintf('%-28s %10.4f %10.1f %10.3f\n', [atype,'-',stype], mean(mse_s), gain, res.offload_rate);
end
fprintf('[EXP 8] Done.\n');
end


% =========================================================================
% EXPERIMENT 9: Latency sensitivity (Table 11)
% =========================================================================
function exp_latency_sensitivity(cfg)

fprintf('\n[EXP 9] Latency Sensitivity\n');
data   = load_dataset(cfg.train_dataset, cfg);
pos_tr = data.positions(data.train_idx,:);
pos_te = data.positions(data.test_idx, :);
srv_tr = server_predictor_init(pos_tr, cfg);

[agent,~,~] = train_agent(pos_tr, srv_tr, cfg, 'DuelingDQN', 'L2', 1);

fprintf('%-20s %12s %12s %12s\n', 'Delta (intervals)','MSE','Gain(%)','OR');
fprintf('%s\n', repmat('-', 58, 1));

for delta = cfg.latency_deltas
    srv_te_d = server_predictor_stale(pos_te, delta, cfg);
    res      = evaluate_agent(agent, pos_te, srv_te_d, cfg);
    fprintf('%-20.1f %12.5f %12.1f %12.3f\n', delta, res.mse_rl, res.gain_rl, res.offload_rate);
end
fprintf('[EXP 9] Done.\n');
end


% =========================================================================
%  HELPER UTILITIES
% =========================================================================

function mse_row = evaluate_multistep(agent, positions, server, cfg, horizon)
% Approximate h-step MSE by compounding single-step predictions.

N  = size(positions, 1) - horizon;
H  = cfg.H;
lms = lms_predictor_init(H, cfg.lms_mu, cfg.lms_n_axes);
for t = 1:H; [~,lms] = lms_predict(lms, positions(t,:)); end

e_rl = zeros(N,1); e_cpo = zeros(N,1);
e_sml = zeros(N,1); e_lml = zeros(N,1); e_genie = zeros(N,1);

for t = H+1:N
    p_cur = positions(t,:); p_fut = positions(t+horizon,:);
    [p_loc,lms_tmp] = lms_predict(lms, p_cur); lms = lms_tmp;
    C_n  = 0.5;
    s_n  = build_state(lms, p_loc, C_n, agent.state_type, cfg);
    [Q,~] = qnetwork_forward(agent.net_online, s_n);
    [~,ai] = max(Q); a_n = ai-1;
    p_srv  = get_server_prediction(server, t);
    p_hat  = p_loc + (a_n==1)*(p_srv - p_loc);   % blend for multi-step
    % compound horizon
    for k = 2:horizon
        p_hat = p_hat + (p_hat - p_loc);   % drift approximation
    end
    e_rl(t-H)    = sum((p_fut - p_hat).^2);
    e_cpo(t-H)   = sum((p_fut - ((rand<cfg.zeta)*p_srv + (rand>=cfg.zeta)*p_loc)).^2);
    e_sml(t-H)   = sum((p_fut - p_loc).^2);
    e_lml(t-H)   = sum((p_fut - p_srv).^2);
    e_genie(t-H) = min(e_sml(t-H), e_lml(t-H));
end
mse_row = [mean(e_rl), mean(e_cpo), mean(e_sml), mean(e_lml), mean(e_genie)];
end


function server_stale = server_predictor_stale(positions, delta, cfg)
% SERVER_PREDICTOR_STALE  Server returns prediction with delta-step delay.
% When delta > 0, the prediction used at epoch t is from epoch t - ceil(delta).

server_stale = server_predictor_init(positions, cfg);
delay = ceil(delta);
if delay > 0
    preds = server_stale.predictions;
    server_stale.predictions = [repmat(preds(1,:), delay, 1); preds(1:end-delay,:)];
end
end


function save_figure(fig, fname, cfg)
% SAVE_FIGURE  Save a figure to the results/figures directory.

if ~cfg.save_figures, return; end
outpath = fullfile(cfg.fig_dir, [fname, '.', cfg.fig_format]);
saveas(fig, outpath);
fprintf('[Figure] Saved: %s\n', outpath);
close(fig);
end
