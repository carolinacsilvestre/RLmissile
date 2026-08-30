function obs = env_obs(env, cfg)
dx = env.xt - env.xm;  dy = env.yt - env.ym;
obs.R   = hypot(dx, dy);
obs.lam = atan2(dy, dx);
obs.Vc  = cfg.mslV*cos(env.gm-obs.lam) - cfg.tgtV*cos(env.gt-obs.lam);
obs.lamdot = (cfg.tgtV*sin(env.gt-obs.lam) - cfg.mslV*sin(env.gm-obs.lam)) / max(obs.R,1e-6);

if obs.Vc > 1e-6
    obs.tgo = min(obs.R/obs.Vc, cfg.tMax);
else
    obs.tgo = cfg.tMax;
end
obs.look  = mod(env.gm-obs.lam+pi, 2*pi) - pi;
obs.aNorm = env.aAch/cfg.aMax;

raw = [ obs.lamdot*min(obs.tgo, cfg.tgoCap)
        obs.tgo
        obs.look
        obs.aNorm
        env.atEst/cfg.accelNorm
        env.atRateEst/cfg.jerkNorm ];

x = 2*(raw - cfg.lo)./(cfg.hi - cfg.lo) - 1;
obs.x = max(-1, min(1, x));
end
