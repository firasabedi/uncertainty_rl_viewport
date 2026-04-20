function [proxy, ensemble] = ensemble_proxy_init(H, mu, n_axes, n_models)
% ENSEMBLE_PROXY_INIT  Initialise an ensemble of LMS predictors for the
%                      ensemble-disagreement ablation (Section 5.4.1).
%
%   [proxy, ensemble] = ensemble_proxy_init(H, mu, n_axes, n_models)
%
%   Creates n_models LMS predictors with slightly perturbed step sizes.
%   The ensemble-disagreement proxy is the variance of their predictions,
%   providing a more principled (but computationally expensive) reliability
%   signal compared to the single-model variance proxy.
%
%   Inputs:
%     H        - history window length
%     mu       - base LMS step size
%     n_axes   - number of position axes (3)
%     n_models - number of ensemble members (3 in paper)
%
%   Outputs:
%     proxy    - scalar proxy value (set to 0 at init)
%     ensemble - cell array of n_models LMS structs
% -------------------------------------------------------------------------

ensemble = cell(n_models, 1);
mu_perturb = linspace(mu * 0.8, mu * 1.2, n_models);

for k = 1:n_models
    ensemble{k} = lms_predictor_init(H, mu_perturb(k), n_axes);
end

proxy = 0;
end


function [proxy, ensemble] = ensemble_proxy_update(ensemble, p_new)
% ENSEMBLE_PROXY_UPDATE  Update all ensemble members and compute disagreement.
%
%   [proxy, ensemble] = ensemble_proxy_update(ensemble, p_new)
%
%   Inputs:
%     ensemble - cell array of LMS structs (from ensemble_proxy_init)
%     p_new    - [1 x n_axes] new observed head position
%
%   Outputs:
%     proxy    - scalar ensemble disagreement (variance of predictions)
%     ensemble - updated ensemble
% -------------------------------------------------------------------------

n_models = length(ensemble);
preds    = zeros(n_models, size(p_new, 2));

for k = 1:n_models
    [preds(k,:), ensemble{k}] = lms_predict(ensemble{k}, p_new);
end

% Ensemble disagreement = variance of predictions across models
% Scalar: pool variance across models and axes
proxy = var(preds(:));
end


function proxy = get_reliability_proxy(lms, ensemble, proxy_type)
% GET_RELIABILITY_PROXY  Unified interface for all proxy variants.
%
%   proxy = get_reliability_proxy(lms, ensemble, proxy_type)
%
%   Inputs:
%     lms        - LMS predictor struct
%     ensemble   - ensemble cell array (only used when proxy_type='ensemble')
%     proxy_type - 'variance' | 'mae' | 'none' | 'ensemble'
%
%   Output:
%     proxy - scalar reliability signal appended to state vector
% -------------------------------------------------------------------------

switch lower(proxy_type)
    case 'variance'
        proxy = compute_variance_proxy(lms);

    case 'mae'
        proxy = compute_mae_proxy(lms);

    case 'none'
        proxy = 0;

    case 'ensemble'
        if isempty(ensemble)
            warning('Ensemble not initialised. Falling back to variance proxy.');
            proxy = compute_variance_proxy(lms);
        else
            % Return last computed ensemble disagreement
            % (updated externally via ensemble_proxy_update)
            proxy = var(cellfun(@(m) norm(m.buffer(end,:)), ensemble));
        end

    otherwise
        error('Unknown proxy_type: %s. Use variance|mae|none|ensemble.', proxy_type);
end
end
