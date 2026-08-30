function verify_plant()
% Plant-layer checks. Several run against the non-manoeuvring target
% (tgtAMax = 0), where the correct answer is known analytically
fprintf('\n=============== PLANT CHECKS ===============\n');
nf = 0;

% collision triangle: a perfect lead gives zero LOS rate at launch
c = cfg_default(); c.tgtAMax = 0; c.he0 = [0 0];
rng(1); worst = 0;
for i = 1:400
    [~, obs] = env_reset(c);
    worst = max(worst, abs(obs.lamdot));
end
nf = nf + rep(1,'lead angle gives lamdot = 0', worst < 1e-12, ...
              sprintf('max |lamdot| = %.2e rad/s', worst));

% analytic kinematics agree with finite differences
c = cfg_default(); c.dt = 1e-4; c.nSub = 1; c.dtInt = 1e-4;
rng(2); eR = 0; eL = 0;
for i = 1:50
    [env, o1] = env_reset(c);
    [~, o2] = env_step(env, 0, c);
    eR = max(eR, abs((o2.R-o1.R)/c.dtInt + o1.Vc)/max(abs(o1.Vc),1));
    dl = mod(o2.lam-o1.lam+pi,2*pi)-pi;
    eL = max(eL, abs(dl/c.dtInt - o1.lamdot)/max(abs(o1.lamdot),1e-6));
end
nf = nf + rep(2,'Vc and lamdot vs finite difference', eR < 1e-3 && eL < 1e-2, ...
              sprintf('rel err Vc %.1e, lamdot %.1e', eR, eL));

% zero command on a perfect lead must intercept exactly
c = cfg_default(); c.tgtAMax = 0; c.he0 = [0 0];
rng(3); worst = 0;
for i = 1:200
    [env, ~] = env_reset(c); done = false; k = 0;
    while ~done && k < c.maxSteps, k = k+1; [env,~,done] = env_step(env,0,c); end
    worst = max(worst, env.miss);
end
nf = nf + rep(3,'zero command on perfect lead', worst < 1e-6, ...
              sprintf('max miss = %.2e m', worst));

% PN intercepts a non-manoeuvring target
c = cfg_default(); c.tgtAMax = 0; c.N = 3;
s = evaluate(c, mlp_init(c.nx,c.nHidden), 300, c.repSeeds);
nf = nf + rep(4,'PN N=3 vs straight target', s.hit >= 99.0, ...
              sprintf('hit %.1f%%, median %.3f m', s.hit, s.med));

% P5 -- integration-step convergence at the operating configuration
c1 = cfg_default();
c2 = c1; c2.nSub = c1.nSub*2; c2.dtInt = c2.dt/c2.nSub;
a = evaluate(c1, mlp_init(c1.nx,c1.nHidden), 200, c1.repSeeds);
b = evaluate(c2, mlp_init(c2.nx,c2.nHidden), 200, c2.repSeeds);
relc = abs(a.hit-b.hit)/max(a.hit,1);
nf = nf + rep(5,'halving dtInt barely moves the answer', relc < 0.05, ...
              sprintf('hit %.1f%% -> %.1f%% (%.1f%% change)', a.hit, b.hit, 100*relc));

% analytic LOS rate has no angle-wrap spikes
c = cfg_default(); rng(6);
ep = run_episode(c, mlp_init(c.nx,c.nHidden));
ld = ep.log.lamdot;
if numel(ld) > 8, jump = max(abs(diff(ld(1:end-3)))); else, jump = inf; end
nf = nf + rep(6,'no LOS-rate wrap spikes', jump < 0.5, ...
              sprintf('max |d(lamdot)| = %.4f rad/s', jump));

% a zero-output network is exactly PN
c = cfg_default(); net = mlp_init(c.nx,c.nHidden); rng(7); worst = 0;
for i = 1:200
    [env, obs] = env_reset(c);
    for k = 1:randi([1 30])
        u = policy_action(net,obs,c); [env,obs,d] = env_step(env,u,c);
        if d, break, end
    end
    upn = max(-c.aMax, min(c.aMax, c.N*obs.Vc*obs.lamdot));
    worst = max(worst, abs(policy_action(net,obs,c) - upn));
end
nf = nf + rep(7,'zero residual reproduces PN', worst == 0, ...
              sprintf('max |u - u_PN| = %.2e m/s^2', worst));

fprintf('===========================================\n');
if nf == 0, fprintf('  ALL PLANT CHECKS PASSED\n\n');
else, fprintf('  %d CHECK(S) FAILED\n\n', nf); end
end

function bad = rep(n, name, pass, extra)
if pass, tag = 'PASS'; bad = 0; else, tag = '**FAIL**'; bad = 1; end
fprintf('  [P%d] %-38s %-9s %s\n', n, name, tag, extra);
end
