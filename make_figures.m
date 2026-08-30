function make_figures(file)
% Report figures. Works with result_seed*.mat (new) or hdp_seed1_trust.mat (old run).
if nargin < 1, file = 'result_seed1.mat'; end
R = load_result(file);
cfg = R.cfg;

% --- Fig 1: learning effect -------------------------------------------
figure('Name','learning effect');
plot(R.it, R.hit, '-o', 'LineWidth',1.4); hold on
plot([min(R.it) max(R.it)], [R.pn.hit R.pn.hit], 'k--', 'LineWidth',1.2);
grid on; xlabel('policy iteration'); ylabel('hit rate [%]');
ylim([floor(min([R.hit(:);R.pn.hit])/5)*5, ceil(max(R.hit)/5)*5]);
legend('residual HDP','PN baseline','Location','southeast');

% --- Fig 2: critic quality and accepted steps -------------------------
figure('Name','training diagnostics');
subplot(2,1,1);
plot(R.it, R.R2, '-s', 'LineWidth',1.4); hold on
plot([min(R.it) max(R.it)], [cfg.minR2 cfg.minR2], 'r--');
grid on; ylabel('critic held-out R^2');
ylim([min(0.55,min(R.R2))-0.05, 1]);
subplot(2,1,2);
stem(R.it, R.accepted, 'filled'); grid on
xlabel('policy iteration'); ylabel('accepted step size');

% --- Fig 3: miss distribution (same episodes, paired) -----------------
figure('Name','miss distribution');
m0 = sort(R.pn.miss); m1 = sort(R.best.miss);
p = (1:numel(m0))/numel(m0);
semilogx(max(m0,1e-2), 100*p, 'LineWidth',1.5); hold on
semilogx(max(m1,1e-2), 100*p, 'LineWidth',1.5);
plot([cfg.Rhit cfg.Rhit], [0 100], 'k--'); grid on
xlabel('miss distance [m]'); ylabel('cumulative [%]');
legend('PN','residual HDP','R_{hit}','Location','southeast');

% --- Fig 4: an engagement PN LOSES and the learned policy WINS --------
net0 = mlp_init(cfg.nx, cfg.nHidden);   % zero weights = PN
[epPN, epRL] = find_episode(cfg, net0, R.net);
nTrim = 30;                               % drop terminal 1/R blow-up in lambda-dot

figure('Name','trajectory');
subplot(3,1,1);
plot(epPN.log.xm/1e3, epPN.log.ym/1e3, 'LineWidth',1.4); hold on
plot(epRL.log.xm/1e3, epRL.log.ym/1e3, 'LineWidth',1.4);
plot(epRL.log.xt/1e3, epRL.log.yt/1e3, 'k--', 'LineWidth',1.1);
plot(epRL.log.xm(end)/1e3, epRL.log.ym(end)/1e3, 'ko', 'MarkerFaceColor','k','MarkerSize',5);
daspect([1 1 1]);
allx=[epPN.log.xm;epRL.log.xm;epRL.log.xt]/1e3; ally=[epPN.log.ym;epRL.log.ym;epRL.log.yt]/1e3;
mx=0.05*max(max(allx)-min(allx), max(ally)-min(ally))+1e-3;
xlim([min(allx)-mx max(allx)+mx]); ylim([min(ally)-mx max(ally)+mx]);
grid on; xlabel('x [km]'); ylabel('y [km]');
title(sprintf('PN miss %.1f m  |  HDP miss %.1f m  (R_{hit} = %g m)', ...
      epPN.miss, epRL.miss, cfg.Rhit));
legend('PN','HDP','target','intercept','Location','best');

subplot(3,1,2);
plot(epPN.log.t, epPN.log.a/9.81, 'LineWidth',1.4); hold on
plot(epRL.log.t, epRL.log.a/9.81, 'LineWidth',1.4);
plot(epRL.log.t, epRL.log.res/9.81, ':', 'LineWidth',1.6);
plot(xlim, [1 1]*cfg.aMax/9.81, 'k:'); plot(xlim, -[1 1]*cfg.aMax/9.81, 'k:');
grid on; ylabel('command [g]');
legend('PN','HDP','learned residual','Location','best');

subplot(3,1,3);
kP = max(1,numel(epPN.log.t)-nTrim); kR = max(1,numel(epRL.log.t)-nTrim);
plot(epPN.log.t(1:kP), epPN.log.lamdot(1:kP), 'LineWidth',1.4); hold on
plot(epRL.log.t(1:kR), epRL.log.lamdot(1:kR), 'LineWidth',1.4);
plot(xlim, [0 0], 'k:');
grid on; xlabel('t [s]'); ylabel('LOS rate [rad/s]');
title(sprintf('final %d samples omitted: \\lambda-dot \\propto 1/R diverges at intercept', nTrim));

% --- Fig 5: what drives the learned residual --------------------------
[Z, U, names, A] = collect_residual_data(cfg, R.net, 60);
Us = (U-mean(U))/max(std(U),eps);
beta = [Z ones(size(Z,1),1)] \ Us;
fit  = corrcoef([Z ones(size(Z,1),1)]*beta, Us);

figure('Name','what drives the learned residual');
b = beta(1:numel(names));
bar(1:numel(b), b, 0.6); grid on; hold on
plot(xlim, [0 0], 'k-');
set(gca, 'XTick', 1:numel(names), 'XTickLabel', names, ...
    'TickLabelInterpreter', 'latex');
ylabel('standardised regression weight');
title({sprintf('residual vs observation   (linear fit R^2 = %.2f)', fit(1,2)^2), ...
       'APN feedforward would put all weight on the manoeuvre level'});

ca = corrcoef(Z(:,5), U);  cr = corrcoef(Z(:,6), U);
fprintf('\nPN %.2f%%  |  HDP best %.2f%%  |  HDP final %.2f%%\n', ...
        R.pn.hit, R.best.hit, R.final.hit);
fprintf('residual RMS %.3f g of %.1f g authority\n', ...
        R.best.resRms/9.81, cfg.aMax/9.81);
fprintf('marginal corr(residual, a_t) = %+.3f ; corr(residual, da_t/dt) = %+.3f\n', ...
        ca(1,2), cr(1,2));
fprintf('standardised weights:');
for i = 1:numel(names), fprintf('  %s %+.3f', names{i}, b(i)); end
fprintf('\n');

% physical projection onto the two manoeuvre channels (Eq. 13 of the report)
M  = [A(:,1) A(:,2) ones(size(A,1),1)];
kk = M \ U;
rk = corrcoef(M*kk, U);
w  = 2*pi/cfg.weaveT;
fprintf('\nphysical projection   u_res ~ k1*a_t + k2*(da_t/dt) + c\n');
fprintf('  k1 = %+.3f          (APN prescribes N/2 = %.2f)\n', kk(1), cfg.N/2);
fprintf('  k2 = %+.3f s        (APN prescribes 0)\n', kk(2));
fprintf('  c  = %+.3f m/s^2    two-channel R2 = %.3f\n', kk(3), rk(1,2)^2);
fprintf('  compensator k1 + k2*s at omega = %.4f rad/s:\n', w);
fprintf('     phase lead %.1f deg, magnitude %.3f\n', ...
        atan2(kk(2)*w, kk(1))*180/pi, hypot(kk(1), kk(2)*w));
fprintf('  autopilot phase lag arctan(omega*tau) = %.1f deg\n', atan(w*cfg.tau)*180/pi);
end

% ---- find an engagement PN loses and the learned policy wins ----------
function [epPN, epRL] = find_episode(cfg, net0, net)
bestGap = -inf; epPN = []; epRL = [];
for s = 1:300
    rng(s); a = run_episode(cfg, net0);
    rng(s); b = run_episode(cfg, net);
    if ~a.hit && b.hit
        epPN = a; epRL = b; return
    end
    gap = a.miss - b.miss;
    if gap > bestGap, bestGap = gap; epPN = a; epRL = b; end
end
end

% ---- observations and residual under the trained policy ---------------
function [Z, U, names, A] = collect_residual_data(cfg, net, nEp)
names = {'drift','$t_{\mathrm{go}}$','$L$','$a_{\mathrm{ach}}$', ...
         '$\hat{a}_{t\perp}$','$\dot{\hat{a}}_{t\perp}$'};
X = zeros(cfg.nx, 0); U = []; A = zeros(0,2);
rng(77);
for i = 1:nEp
    [env, obs] = env_reset(cfg); done = false; k = 0;
    while ~done && k < cfg.maxSteps
        k = k+1;
        [u,~,~,uRes] = policy_action(net, obs, cfg);
        X(:,end+1) = obs.x; U(end+1) = uRes;
        A(end+1,:) = [env.atEst, env.atRateEst];
        [env, obs, done] = env_step(env, u, cfg);
    end
end
U = U(:);
Z = X.';
Z = (Z - mean(Z)) ./ max(std(Z), 1e-9);
end

function R = load_result(file)
S = load(file);
R.cfg = cfg_default();
if size(S.curve,2) > 7            % legacy 17-column curve
    c = S.curve;
    R.it=c(:,1); R.R2=c(:,2); R.accepted=c(:,7); R.hit=c(:,12);
    R.pn=S.s0Rep; R.best=S.sBest; R.final=S.sFinal;
    R.net=struct('nx',S.ctrl.fa.nx,'nh',S.ctrl.fa.nh,'w',S.ctrl.fa.w);
else
    R.cfg = S.cfg;
    c = S.curve;
    R.it=c(:,1); R.R2=c(:,2); R.accepted=c(:,4); R.hit=c(:,5);
    R.pn=S.pn; R.best=S.bestStats; R.final=S.final;
    R.net=S.best.net;
end
end
