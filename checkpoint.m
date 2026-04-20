function lms = lms_predictor_init(H, mu, n_axes)
% LMS_PREDICTOR_INIT  Initialise a Least Mean Squares adaptive predictor.
%
%   lms = lms_predictor_init(H, mu, n_axes)
%
%   Inputs:
%     H       - history window length
%     mu      - LMS step size (learning rate)
%     n_axes  - number of independent axes (3 for x,y,z)
%
%   Output:
%     lms struct with fields:
%       .W        [H x n_axes] weight matrix (one vector per axis)
%       .H        history length
%       .mu       step size
%       .n_axes   number of axes
%       .buffer   [H x n_axes] rolling position buffer
%       .residuals [H x n_axes] recent prediction residuals
%
% UPDATE RULE (Eqs. 3-5 in paper):
%   p_hat_{t+1} = w_t' * p_{t-H+1:t}
%   e_t         = p_{t+1} - p_hat_{t+1}
%   w_{t+1}     = w_t + mu * e_t * p_{t-H+1:t}
% -------------------------------------------------------------------------

lms.W         = zeros(H, n_axes);   % initialise weights to zero
lms.H         = H;
lms.mu        = mu;
lms.n_axes    = n_axes;
lms.buffer    = zeros(H, n_axes);   % rolling window of past positions
lms.residuals = zeros(H, n_axes);   % rolling window of past residuals
lms.step      = 0;                  % internal step counter
end


function [p_hat, lms] = lms_predict(lms, p_new)
% LMS_PREDICT  Generate a one-step-ahead prediction and update weights.
%
%   [p_hat, lms] = lms_predict(lms, p_new)
%
%   Inputs:
%     lms   - LMS struct (from lms_predictor_init or prior lms_predict call)
%     p_new - [1 x n_axes] current observed head position
%
%   Outputs:
%     p_hat - [1 x n_axes] predicted next head position
%     lms   - updated LMS struct
% -------------------------------------------------------------------------

H      = lms.H;
mu     = lms.mu;
n_axes = lms.n_axes;

% 1. Shift buffer: discard oldest, append newest
lms.buffer = [lms.buffer(2:end, :); p_new];

% 2. Predict: p_hat = W' * buffer  (Eq. 3)
p_hat = sum(lms.W .* lms.buffer, 1);   % [1 x n_axes]

% 3. Compute residual using PREVIOUS ground truth (Eq. 4)
%    On first H steps, ground truth not yet available — skip update
lms.step = lms.step + 1;
if lms.step > H
    % Residual of PREVIOUS prediction vs. CURRENT ground truth
    e = p_new - p_hat;   % [1 x n_axes]  (Eq. 4)

    % 4. Update weights (Eq. 5): w_{t+1} = w_t + mu * e * buffer
    lms.W = lms.W + mu * (lms.buffer .* e);   % broadcast e across rows

    % 5. Update residuals buffer
    lms.residuals = [lms.residuals(2:end, :); e];
end
end


function sigma2 = compute_variance_proxy(lms)
% COMPUTE_VARIANCE_PROXY  σ²_n = Var of recent prediction residuals.
%
%   sigma2 = compute_variance_proxy(lms)
%
%   Computes the scalar variance proxy (Eq. 6 in paper) by pooling
%   residuals across all axes and computing the variance over H samples.
%
%   Output:
%     sigma2 - scalar, the variance proxy σ²_n
% -------------------------------------------------------------------------

residuals_flat = lms.residuals(:);       % [H*n_axes x 1]
if var(residuals_flat) == 0 && all(residuals_flat == 0)
    sigma2 = 0;   % early steps: not yet populated
else
    sigma2 = var(residuals_flat);
end
end


function mae_proxy = compute_mae_proxy(lms)
% COMPUTE_MAE_PROXY  MAE proxy for ablation comparison.
%
%   mae_proxy = compute_mae_proxy(lms)
%
%   Returns mean absolute error of recent residuals (pooled across axes).
% -------------------------------------------------------------------------

mae_proxy = mean(abs(lms.residuals(:)));
end


function lms_noisy = add_lms_noise(lms_clean, noise_std)
% ADD_LMS_NOISE  Inject Gaussian noise into LMS residuals to simulate
%               a degraded local predictor (Section 5.9 of the paper).
%
%   lms_noisy = add_lms_noise(lms_clean, noise_std)
% -------------------------------------------------------------------------

lms_noisy           = lms_clean;
lms_noisy.residuals = lms_clean.residuals + noise_std * randn(size(lms_clean.residuals));
end
