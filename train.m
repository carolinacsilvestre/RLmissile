function result = train(cfg)
if nargin < 1, cfg = cfg_default(); end
rng(cfg.seed);

net = mlp_init(cfg.nx, cfg.nHidden);   % zero output layer => u is exactly PN
s0  = evaluate(cfg, net, cfg.nEval, cfg.repSeeds);
sel0= evaluate(cfg, net, cfg.nEval, cfg.selSeeds);
fprintf('iter  0 | PN baseline | hit %5.1f%% | J %7.3f\n', s0.hit, s0.J);

[critic, fit] = fit_critic(cfg, net, [], cfg.nTrainEp, cfg.nValEp, cfg.epochs0, 1000*cfg.seed);
fprintf('critic (%s): held-out R2 %.3f\n', cfg.criticMethod, fit.R2);
if ~fit.passed
    error('train:criticGate','held-out R2 %.3f below %.3f', fit.R2, cfg.minR2);
end

best = struct('hit',sel0.hit, 'net',net, 'iter',0);
curve = nan(cfg.nIter+1, 7);   % [iter R2 trustRms accepted repHit repJ resRms]
curve(1,:) = [0 fit.R2 0 0 s0.hit s0.J 0];

for it = 1:cfg.nIter
    if it > 1
        [critic, fit] = fit_critic(cfg, net, critic, round(cfg.nTrainEp/3), ...
                                   round(cfg.nValEp/2), cfg.epochsRef, 1000*cfg.seed+it);
        if ~fit.passed
            warning('critic R2 %.3f below gate, stopping', fit.R2);
            curve = curve(1:it,:); break
        end
    end

    [envs, obss] = collect_states(cfg, net, 3000+it);
    [dw, gs] = actor_update(cfg, net, critic, envs, obss);

    accSeeds = 4100 + 10*it + (0:2);
    base = evaluate(cfg, net, cfg.nAccept, accSeeds);
    accepted = 0;
    for f = cfg.lineSearch
        cand = net; cand.w = net.w + f*dw;
        sc = evaluate(cfg, cand, cfg.nAccept, accSeeds);
        if sc.J >= base.J + cfg.minImprove && sc.hit >= base.hit - cfg.hitDropTol
            net = cand; accepted = f; break
        end
    end

    sRep = evaluate(cfg, net, cfg.nEval, cfg.repSeeds);
    sSel = evaluate(cfg, net, cfg.nEval, cfg.selSeeds);
    if sSel.hit > best.hit, best = struct('hit',sSel.hit,'net',net,'iter',it); end

    curve(it+1,:) = [it fit.R2 gs.rms/9.81 accepted sRep.hit sRep.J sRep.resRms/9.81];
    fprintf('iter %2d | R2 %.3f | step %.3f g x %.4g | hit %5.1f%% | residual %.2f g\n', ...
            it, fit.R2, gs.rms/9.81, accepted, sRep.hit, sRep.resRms/9.81);
end

sFinal = evaluate(cfg, net, cfg.nEval, cfg.repSeeds);
sBest  = evaluate(cfg, best.net, cfg.nEval, cfg.repSeeds);
fprintf('\nPN %.1f%%  ->  final %.1f%%  |  best (iter %d) %.1f%%\n', ...
        s0.hit, sFinal.hit, best.iter, sBest.hit);

result = struct('cfg',cfg, 'net',net, 'best',best, 'critic',critic, 'curve',curve, ...
                'pn',s0, 'final',sFinal, 'bestStats',sBest, 'fit',fit);
save(sprintf('result_seed%d.mat', cfg.seed), '-struct', 'result');
end

% ---- states visited by the frozen greedy policy -------------------------
function [envs, obss] = collect_states(cfg, net, seed)
old = rng; rng(seed);
envs = {}; obss = {}; q = 0;
for ep = 1:cfg.stateEp
    [env, obs] = env_reset(cfg);
    k = 0; done = false;
    while ~done && k < cfg.maxSteps
        k = k+1;
        if mod(k-1,2) == 0
            q = q+1; envs{q} = env; obss{q} = obs;
        end
        u = policy_action(net, obs, cfg);
        [env, obs, done] = env_step(env, u, cfg);
    end
end
if q > cfg.maxStates
    keep = randperm(q, cfg.maxStates);
    envs = envs(keep); obss = obss(keep);
end
rng(old);
end
