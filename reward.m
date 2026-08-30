function r = reward(obs, u, done, miss, cfg)
drift = obs.lamdot*min(obs.tgo, cfg.tgoCap);
dn = min(abs(drift)/cfg.driftScale, cfg.driftClip);
un = u/cfg.aMax;
r = -(cfg.cTrack*dn + cfg.cEffort*un^2)*cfg.dt;

if done
    near = cfg.cNear*min(miss, cfg.rKnee)/cfg.rKnee;
    far  = cfg.cFar *max(0, min(miss,cfg.missCap)-cfg.rKnee)/(cfg.missCap-cfg.rKnee);
    r = r - (near + far);
end
end
