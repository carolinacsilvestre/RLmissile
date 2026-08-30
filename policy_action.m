function [u, duDy, uBase, uRes] = policy_action(net, obs, cfg)
y = mlp_eval(net, obs.x);
uBase = max(-cfg.aMax, min(cfg.aMax, cfg.N*obs.Vc*obs.lamdot));
u     = max(-cfg.aMax, min(cfg.aMax, uBase + cfg.aMax*tanh(y)));
duDy  = cfg.aMax*(1-tanh(y)^2);
uRes  = u - uBase;
end
