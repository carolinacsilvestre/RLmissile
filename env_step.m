function [env, obs, done] = env_step(env, aCmd, cfg)
dt = cfg.dtInt;
aCmd = max(-cfg.aMax, min(cfg.aMax, aCmd));
done = false;
o0 = env_obs(env, cfg);  t0 = env.t;

for k = 1:cfg.nSub
    vmx = cfg.mslV*cos(env.gm); vmy = cfg.mslV*sin(env.gm);
    vtx = cfg.tgtV*cos(env.gt); vty = cfg.tgtV*sin(env.gt);
    rx = env.xt-env.xm; ry = env.yt-env.ym;
    vx = vtx-vmx;       vy = vty-vmy;

    tca = -(rx*vx + ry*vy)/max(vx*vx+vy*vy, 1e-9);
    if tca >= 0 && tca <= dt
        env.miss = hypot(rx+vx*tca, ry+vy*tca);
        env.hit  = env.miss < cfg.Rhit;
        env.t    = env.t + tca;
        done = true; break
    end

    if cfg.tau > 0
        env.aAch = env.aAch + (aCmd-env.aAch)*dt/cfg.tau;
    else
        env.aAch = aCmd;
    end
    at = cfg.tgtAMax*env.sgn*sin(2*pi*env.t/cfg.weaveT);

    env.gm = wrapLocal(env.gm + (env.aAch/cfg.mslV)*dt);
    env.gt = wrapLocal(env.gt + (at/cfg.tgtV)*dt);
    env.xm = env.xm + cfg.mslV*cos(env.gm)*dt;
    env.ym = env.ym + cfg.mslV*sin(env.gm)*dt;
    env.xt = env.xt + cfg.tgtV*cos(env.gt)*dt;
    env.yt = env.yt + cfg.tgtV*sin(env.gt)*dt;
    env.t  = env.t + dt;
end

if ~done
    o1 = env_obs(env, cfg);
    h  = max(env.t-t0, 1e-6);
    lamddot = (o1.lamdot - o0.lamdot)/h;
    raw = o1.R*lamddot - 2*o1.Vc*o1.lamdot + env.aAch*cos(o1.look);
    cap = cfg.accelCap*cfg.accelNorm;
    raw = max(-cap, min(cap, raw));
    a = h/(cfg.accelTau + h);
    old = env.atEst;
    env.atEst = old + a*(raw-old);
    rcap = cfg.accelCap*cfg.jerkNorm;
    env.atRateEst = max(-rcap, min(rcap, (env.atEst-old)/h));
end

obs = env_obs(env, cfg);

if ~done && (env.t >= cfg.tMax || obs.Vc <= 0)
    env.miss = obs.R; env.hit = false; done = true;
end
end

function a = wrapLocal(a)
a = mod(a+pi, 2*pi) - pi;
end
