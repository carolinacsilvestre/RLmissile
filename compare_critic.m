function compare_critic(nTrain, nVal, nEpoch)
% Monte Carlo vs TD(0) policy evaluation, same policy and same held-out data.
% No training run: this only fits critics to the PN policy.
% TD is given its best learning rate over a small sweep, so the comparison
% cannot be dismissed as a handicapped baseline.
%
% Run with NO arguments for the reportable numbers. Small sizes mislead:
% MC is the higher-variance estimator and needs the data, so at a few hundred
% episodes it scores BELOW TD and the conclusion inverts.
c0 = cfg_default();
if nargin < 1 || isempty(nTrain), nTrain = c0.nTrainEp; end
if nargin < 2 || isempty(nVal),   nVal   = c0.nValEp;   end
if nargin < 3 || isempty(nEpoch), nEpoch = c0.epochs0;  end

net   = mlp_init(c0.nx, c0.nHidden);   % zero output layer => policy is exactly PN
betas = [0.1 0.3 1.0];
seed  = 1;

c = c0; c.criticMethod = 'mc';
t0 = tic; [~, fMC] = fit_critic(c, net, [], nTrain, nVal, nEpoch, seed);
fprintf('\nMonte Carlo    held-out R2 = %.3f   (%.0f s)\n', fMC.R2, toc(t0));

R2 = zeros(size(betas));
for i = 1:numel(betas)
    c = c0; c.criticMethod = 'td'; c.betaTD = betas(i);
    t0 = tic; [~, fTD] = fit_critic(c, net, [], nTrain, nVal, nEpoch, seed);
    R2(i) = fTD.R2;
    fprintf('TD(0) beta=%.2f held-out R2 = %.3f   (%.0f s)\n', betas(i), R2(i), toc(t0));
end

[bR2, bi] = max(R2);
fprintf('\n  MC    %.3f\n  TD(0) %.3f  (best of beta = %.2f)\n', fMC.R2, bR2, betas(bi));
fprintf('  gate  %.2f  -->  MC %s, TD %s\n', c0.minR2, ...
        tf(fMC.R2>=c0.minR2), tf(bR2>=c0.minR2));
save('compare_critic.mat','fMC','R2','betas','nTrain','nVal','nEpoch','seed');
end

function s = tf(b)
if b, s = 'passes'; else, s = 'BLOCKED'; end
end
