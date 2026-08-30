function [env, obs] = env_reset(cfg)
R0 = cfg.R0(1) + rand*diff(cfg.R0);
gt = cfg.gt0(1) + rand*diff(cfg.gt0);

env.xm = 0; env.ym = 0;
env.xt = R0;  env.yt = 0;
env.gt = gt;

sinL = cfg.tgtV*sin(gt)/cfg.mslV;
he = cfg.he0(1) + rand*diff(cfg.he0);
env.gm = wrapToPiLocal(asin(sinL) + he);

env.t = 0;
env.aAch = 0;
env.sgn = sign(rand-0.5); if env.sgn==0, env.sgn = 1; end
env.atEst = 0; env.atRateEst = 0;
env.miss = NaN; env.hit = false;

obs = env_obs(env, cfg);
end

function a = wrapToPiLocal(a)
a = mod(a+pi, 2*pi) - pi;
end
