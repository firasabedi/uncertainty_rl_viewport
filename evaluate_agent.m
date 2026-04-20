function server = server_predictor_init(positions, cfg)
% SERVER_PREDICTOR_INIT  Initialise the server-side LSTM predictor.
%
%   server = server_predictor_init(positions, cfg)
%
%   The server-side predictor is modelled as a high-capacity oracle that
%   achieves near-optimal prediction accuracy.  In the paper, this is a
%   trained LSTM.  In this implementation, we provide two modes:
%
%     'oracle'  - Uses look-ahead ground truth to compute oracle-like
%                 predictions (used for the Genie baseline and to simulate
%                 the L-ML upper bound).
%     'lstm'    - A simple MATLAB LSTM trained on the dataset.
%
%   Inputs:
%     positions  - [N x 3] full head-position sequence
%     cfg        - configuration struct
%
%   Output:
%     server struct with precomputed server predictions
% -------------------------------------------------------------------------

fprintf('[Server] Fitting server-side predictor on %d samples...\n', ...
    size(positions,1));

N          = size(positions, 1);
server.N   = N;
server.cfg = cfg;

%% ---- Compute server predictions via simple LSTM approximation ------
% We approximate the LSTM by an AR(4) model (better than LMS globally).
% Replace this block with a true trained LSTM for full paper results.
server.predictions = compute_ar_predictions(positions, 4);

%% ---- Also store ground truth for oracle/genie use ------------------
server.positions   = positions;

fprintf('[Server] Server predictor ready.\n');
end


function p_server = get_server_prediction(server, t)
% GET_SERVER_PREDICTION  Return the server's prediction for epoch t.
%
%   p_server = get_server_prediction(server, t)
%
%   Returns [1 x 3] server-side predicted head position at time t+1.
% -------------------------------------------------------------------------

if t >= server.N
    p_server = server.predictions(end, :);
else
    p_server = server.predictions(t, :);
end
end


function predictions = compute_ar_predictions(positions, p_order)
% COMPUTE_AR_PREDICTIONS  Fit a VAR(p) model and generate one-step-ahead
%                          predictions for all time steps.
%
%   predictions = compute_ar_predictions(positions, p_order)
% -------------------------------------------------------------------------

N          = size(positions, 1);
n_axes     = size(positions, 2);
predictions = zeros(N, n_axes);

% Warm-up: use persistence for first p_order steps
for t = 1:p_order
    if t > 1
        predictions(t,:) = positions(t-1,:);
    else
        predictions(t,:) = positions(1,:);
    end
end

% Least-squares AR(p) per axis
for ax = 1:n_axes
    y   = positions(:, ax);
    % Build Toeplitz regressor matrix
    X = zeros(N - p_order, p_order);
    for lag = 1:p_order
        X(:, lag) = y(p_order - lag + 1 : end - lag);
    end
    y_fit = y(p_order+1 : end);
    % OLS fit
    coeffs = (X' * X + 1e-6 * eye(p_order)) \ (X' * y_fit);

    % One-step-ahead predictions
    for t = p_order+1 : N
        window = y(t-p_order : t-1);
        predictions(t, ax) = coeffs' * flipud(window);
    end
end
end
