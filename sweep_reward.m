function sweep_reward(nTrain, nVal, nEpoch)
% Sensitivity of the OBJECTIVE to the discount and the reward shape.
% Each point is one Monte Carlo critic fit to the PN policy -- no training
% runs. Held-out R^2 is scale-invariant, so points stay comparable even
% though the returns themselves are not comparable across reward settings.
% All points share one budget, so differences are meaningful even if no
% single fit is fully converged.
c0 = cfg_default();
if nargin < 1 || isempty(nTrain), nTrain = 1000; end
if nargin < 2 || isempty(nVal),   nVal   = 250;  end
if nargin < 3 || isempty(nEpoch), nEpoch = 80;   end
net = mlp_init(c0.nx, c0.nHidden);      % zero output layer => PN policy
nSteps = 115;                           % mean episode length

% ---- A. discount factor ---------------------------------------------
gam = [0.95 0.99 0.995 0.999];
R2g = zeros(size(gam));
att = gam.^nSteps;                      % terminal signal seen at launch
hor = 1./(1-gam);                       % effective horizon [steps]
fprintf('\n=== A. discount factor (episode is %d steps) ===\n', nSteps);
fprintf('  gamma   horizon   gamma^%d   held-out R2\n', nSteps);
for i = 1:numel(gam)
    c = c0; c.gamma = gam(i);
    [~, f] = fit_critic(c, net, [], nTrain, nVal, nEpoch, 1);
    R2g(i) = f.R2;
    fprintf('  %.3f   %7.0f   %7.3f   %.3f\n', gam(i), hor(i), att(i), R2g(i));
end

% ---- B. terminal-cost knee ------------------------------------------
knee = [4 25; 3 40; 2.5 50; 1.5 100];   % [cNear, rKnee], cFar = 6 - cNear
R2k = zeros(1,size(knee,1)); step = R2k;
fprintf('\n=== B. terminal-cost knee (worst case held at 6) ===\n');
fprintf('  cNear  rKnee   cost step across Rhit   held-out R2\n');
for i = 1:size(knee,1)
    c = c0; c.cNear = knee(i,1); c.rKnee = knee(i,2); c.cFar = 6 - c.cNear;
    step(i) = c.cNear*(15-5)/c.rKnee;   % cost of a 15 m miss vs a 5 m hit
    [~, f] = fit_critic(c, net, [], nTrain, nVal, nEpoch, 1);
    R2k(i) = f.R2;
    fprintf('  %4.1f   %5.0f   %8.2f               %.3f\n', ...
            knee(i,1), knee(i,2), step(i), R2k(i));
end

% ---- C. effort weight ------------------------------------------------
eff = [0.25 0.5 1.0 2.0];
R2e = zeros(size(eff));
fprintf('\n=== C. effort weight ===\n');
fprintf('  cEffort   held-out R2\n');
for i = 1:numel(eff)
    c = c0; c.cEffort = eff(i);
    [~, f] = fit_critic(c, net, [], nTrain, nVal, nEpoch, 1);
    R2e(i) = f.R2;
    fprintf('  %5.2f       %.3f\n', eff(i), R2e(i));
end

save('sweep_reward.mat','gam','R2g','att','hor','knee','R2k','step', ...
     'eff','R2e','nTrain','nVal','nEpoch','nSteps');

figure('Name','objective sensitivity');
subplot(1,3,1);
semilogx(hor, R2g, '-o', hor, att, '-s', 'LineWidth', 1.5); hold on
plot([nSteps nSteps], [0 1], 'k:');
grid on; ylim([0 1]); xlabel('effective horizon 1/(1-\gamma)   [steps]');
legend('critic held-out R^2', '\gamma^{115}', 'episode length', ...
       'Location', 'south');
title('discount');
subplot(1,3,2);
plot(step, R2k, '-o', 'LineWidth',1.5); grid on
xlabel('terminal cost step across R_{hit}'); ylabel('critic held-out R^2');
title('reward steepness');
subplot(1,3,3);
plot(eff, R2e, '-o', 'LineWidth',1.5); grid on
xlabel('effort weight c_{ef}'); ylabel('critic held-out R^2');
title('effort weight');
end
