function [dw, st] = actor_update(cfg, net, critic, envs, obss)
n  = numel(envs);
nw = numel(net.w);
D  = zeros(nw, n); dq = zeros(1, n);

for i = 1:n
    [~, gw] = mlp_eval(net, obss{i}.x);
    [u, duDy] = policy_action(net, obss{i}, cfg);
    dq(i) = dQdu(cfg, net, critic, envs{i}, u);
    D(:,i) = duDy*gw;
end

g = D*dq.'/max(n,1);

proj = D.'*g;
unit = sqrt(mean(proj.^2));
if ~isfinite(unit) || unit < 1e-12
    dw = zeros(size(g));
    st = struct('meanAbsDQ',mean(abs(dq)), 'rms',0);
    return
end

dw = (cfg.trustRms/unit)*g;
if norm(dw) > cfg.maxWeightStep, dw = dw*(cfg.maxWeightStep/norm(dw)); end
for k = 1:3
    a = delta_rms(cfg, net, dw, obss);
    if a <= cfg.trustRms*1.001 || a < 1e-12, break, end
    dw = dw*(cfg.trustRms/a);
end
st = struct('meanAbsDQ',mean(abs(dq)), 'rms',delta_rms(cfg,net,dw,obss));
end

% central difference of the H-step action value 
function d = dQdu(cfg, net, critic, env0, u0)
du = cfg.fdFrac*cfg.aMax;
uL = max(-cfg.aMax, u0-du);
uH = min( cfg.aMax, u0+du);
qL = rollout(cfg, net, critic, env0, uL);
qH = rollout(cfg, net, critic, env0, uH);
d  = (qH-qL)/max(uH-uL, eps);
d  = max(-cfg.gradClip, min(cfg.gradClip, d));
end

function Q = rollout(cfg, net, critic, env0, uFirst)
env = env0; obs = env_obs(env, cfg);
disc = 1; Q = 0; done = false;
for k = 1:cfg.H
    if k == 1, u = uFirst; else, u = policy_action(net, obs, cfg); end
    [env, obs, done] = env_step(env, u, cfg);
    Q = Q + disc*reward(obs, u, done, env.miss, cfg);
    disc = disc*cfg.gamma;
    if done, break, end
end
if ~done, Q = Q + disc*mlp_eval(critic, obs.x); end
end

function z = delta_rms(cfg, net, dw, obss)
n2 = net; n2.w = net.w + dw;
d = zeros(numel(obss),1);
for i = 1:numel(obss)
    d(i) = policy_action(n2, obss{i}, cfg) - policy_action(net, obss{i}, cfg);
end
z = sqrt(mean(d.^2));
end
