function M = compute_metrics(p_true, p_pred, baseline_persist)
% COMPUTE_METRICS  Compute all evaluation metrics reported in the paper.
%
%   M = compute_metrics(p_true, p_pred)
%   M = compute_metrics(p_true, p_pred, baseline_persist)
%
%   Inputs:
%     p_true           - [N x 3] ground-truth head positions
%     p_pred           - [N x 3] predicted head positions
%     baseline_persist - [N x 3] persistence predictions (optional,
%                         required for Skill Score)
%
%   Outputs (struct):
%     M.mse    - Mean Squared Error
%     M.rmse   - Root MSE
%     M.wi     - Willmott's Index of Agreement [37] (Eq. 12)
%     M.ss     - Skill Score vs persistence baseline (Eq. 12)
%     M.apb    - Absolute Percentage Bias % (Eq. 12)
%     M.evs    - Explained Variance Score (Eq. 12)
% =========================================================================

%% ---- Flatten to 1D errors -----------------------------------------
e       = p_true - p_pred;       % [N x 3] errors
e_flat  = e(:);                  % [3N x 1]
p_flat  = p_true(:);
q_flat  = p_pred(:);

%% ---- MSE / RMSE ---------------------------------------------------
M.mse  = mean(sum(e.^2, 2));    % per-sample MSE, averaged (paper metric)
M.rmse = sqrt(M.mse);

%% ---- Willmott's Index of Agreement (Eq. 12 in paper) ---------------
%   WI = 1 - sum((P_i - O_i)^2) / sum((|P_i - O_bar| + |O_i - O_bar|)^2)
O_bar   = mean(p_flat);
num_wi  = sum((q_flat - p_flat).^2);
den_wi  = sum((abs(q_flat - O_bar) + abs(p_flat - O_bar)).^2);
if den_wi < 1e-12
    M.wi = NaN;
else
    M.wi = 1 - num_wi / den_wi;
end

%% ---- Skill Score vs persistence (Eq. 12) ----------------------------
%   SS = 1 - MSE_model / MSE_persistence
if nargin >= 3 && ~isempty(baseline_persist)
    e_pers = p_true - baseline_persist;
    mse_persist = mean(sum(e_pers.^2, 2));
    if mse_persist < 1e-12
        M.ss = NaN;
    else
        M.ss = 1 - M.mse / mse_persist;
    end
else
    % Default: persistence = previous position
    if size(p_true,1) > 1
        p_persist   = [p_true(1,:); p_true(1:end-1,:)];
        e_pers      = p_true - p_persist;
        mse_persist = mean(sum(e_pers.^2, 2));
        M.ss        = 1 - M.mse / max(mse_persist, 1e-12);
    else
        M.ss = NaN;
    end
end

%% ---- Absolute Percentage Bias % (Eq. 12) ----------------------------
%   APB = |sum(P_i - O_i)| / sum(|O_i|) * 100
denom_apb = sum(abs(p_flat));
if denom_apb < 1e-12
    M.apb = NaN;
else
    M.apb = 100 * abs(sum(q_flat - p_flat)) / denom_apb;
end

%% ---- Explained Variance Score (Eq. 12) -----------------------------
%   EVS = 1 - Var(O - P) / Var(O)
var_O  = var(p_flat);
var_eP = var(p_flat - q_flat);
if var_O < 1e-12
    M.evs = NaN;
else
    M.evs = 1 - var_eP / var_O;
end
end


function [U, p_val, r_eff] = mann_whitney_u(errors_local, errors_offloaded)
% MANN_WHITNEY_U  Two-sided Mann-Whitney U test (Section 5.6 of paper).
%
%   [U, p_val, r_eff] = mann_whitney_u(errors_local, errors_offloaded)
%
%   Tests whether locally-processed tasks and offloaded tasks are drawn
%   from the same error distribution.
%
%   Inputs:
%     errors_local    - prediction errors for locally processed tasks
%     errors_offloaded - prediction errors for offloaded tasks
%
%   Outputs:
%     U       - Mann-Whitney U statistic
%     p_val   - two-sided p-value (uses Normal approximation for large N)
%     r_eff   - effect size r = Z / sqrt(N)
% -------------------------------------------------------------------------

n1 = length(errors_local);
n2 = length(errors_offloaded);

% Rank all observations jointly
all_vals = [errors_local(:); errors_offloaded(:)];
[~, sort_idx] = sort(all_vals);
ranks = zeros(n1+n2, 1);
ranks(sort_idx) = 1:n1+n2;

% Handle ties (average ranks)
[unique_vals, ~, ic] = unique(all_vals);
for k = 1:length(unique_vals)
    tied = find(ic == k);
    if length(tied) > 1
        ranks(tied) = mean(ranks(tied));
    end
end

R1 = sum(ranks(1:n1));   % rank sum for group 1
U1 = R1 - n1*(n1+1)/2;
U2 = n1*n2 - U1;
U  = min(U1, U2);

% Normal approximation (valid for n1, n2 > 20)
mu_U    = n1*n2/2;
sigma_U = sqrt(n1*n2*(n1+n2+1)/12);
Z       = (U - mu_U) / sigma_U;
p_val   = 2 * (1 - normcdf(abs(Z)));   % two-sided
r_eff   = abs(Z) / sqrt(n1 + n2);      % Cohen's r

fprintf('[Mann-Whitney] U=%.0f | p=%.2e | r_eff=%.2f\n', U, p_val, r_eff);
end


function print_results_table(all_results, agent_labels, dataset_name)
% PRINT_RESULTS_TABLE  Pretty-print a results table matching paper format.
%
%   Columns: Method | MSE | CI | Gain(%)
% -------------------------------------------------------------------------

fprintf('\n=== Results: %s ===\n', dataset_name);
fprintf('%-25s %10s  %22s  %10s  %10s\n', ...
    'Method', 'MSE', '95% CI', 'Gain(%)', 'OR');
fprintf('%s\n', repmat('-', 1, 80));

for k = 1:length(all_results)
    r   = all_results{k};
    lbl = agent_labels{k};
    fprintf('%-25s %10.5f  (%8.5f--%8.5f)  %10.1f  %10.3f\n', ...
        lbl, r.mse_mean, r.ci_lo, r.ci_hi, r.gain_mean, r.or_mean);
end
fprintf('%s\n', repmat('-', 1, 80));

% Fixed baselines
if ~isempty(all_results)
    r0 = all_results{1};
    fprintf('%-25s %10.5f\n', 'S-ML', r0.mse_sml);
    fprintf('%-25s %10.5f\n', 'L-ML', r0.mse_lml);
    fprintf('%-25s %10.5f\n', 'CPO',  r0.mse_cpo);
    fprintf('%-25s %10.5f\n', 'Genie',r0.mse_genie);
end
end
