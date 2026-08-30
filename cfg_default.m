function cfg = cfg_default()
G = 9.81;

% simulation 
cfg.dt      = 0.05;
cfg.nSub    = 5;
cfg.dtInt   = cfg.dt/cfg.nSub;
cfg.tMax    = 20;
cfg.maxSteps= ceil(cfg.tMax/cfg.dt);

% missile 
cfg.mslV    = 600;
cfg.aMax    = 6*G;           % REPORT: difficulty knob
cfg.tau     = 0.30;          % REPORT: autopilot lag, difficulty knob

% target 
cfg.tgtV    = 300;
cfg.tgtAMax = 5*G;           % REPORT: difficulty knob (0 = straight target)
cfg.weaveT  = 3.0;           % REPORT: difficulty knob

% geometry / termination
cfg.R0      = [3000 6000];
cfg.gt0     = [0.55*pi 1.45*pi];
cfg.he0     = [-10 10]*pi/180;
cfg.Rhit    = 10;

% baseline guidance 
cfg.N       = 2.5;             % REPORT: navigation constant

% state normalisation (frozen: must NOT track tgtAMax during sweeps) 
cfg.tgoCap    = 20;
cfg.accelTau  = 0.10;
cfg.accelCap  = 1.50;
cfg.accelNorm = max(5*G, G);
cfg.jerkNorm  = max(5*G, G)*2*pi/3.0;
cfg.lo = [-0.30; 0; -pi/3; -1; -1.5; -1.5];
cfg.hi = [ 0.30; 8;  pi/3;  1;  1.5;  1.5];
cfg.nx = 6;

% reward (pure cost: r <= 0 everywhere, lecture eq. 11) 
cfg.cTrack     = 1.0;
cfg.cEffort    = 0.5;
cfg.driftScale = 0.15;
cfg.driftClip  = 3.0;
cfg.cNear      = 3.0;
cfg.rKnee      = 40;
cfg.cFar       = 3.0;
cfg.missCap    = 200;

% learning 
cfg.gamma      = 0.995;      % REPORT: sensitivity sweep
cfg.nHidden    = 32;         % REPORT: sensitivity sweep
cfg.H          = max(1,ceil(cfg.tau/cfg.dt));  % REPORT: H=1 is the lecture one-step gradient
cfg.fdFrac     = 1e-3;
cfg.gradClip   = 0.10;
cfg.trustRms   = 0.10*G;     % REPORT: headline sensitivity sweep
cfg.maxWeightStep = 0.05;
cfg.lineSearch = [1 0.5 0.25 0.125 0.0625];
cfg.hitDropTol = 1.0;
cfg.minImprove = 1e-3;

% critic fitting
cfg.criticMethod = 'mc';     % REPORT: 'mc' or 'td' (lecture: Monte Carlo vs TD)
cfg.nTrainEp   = 2000;
cfg.nValEp     = 400;
cfg.stride     = 2;
cfg.epochs0    = 200;
cfg.epochsRef  = 25;
cfg.batchSize  = 512;
cfg.learnRate  = 2e-3;
cfg.weightDecay= 1e-6;
cfg.fitClip    = 5.0;
cfg.patience   = 12;
cfg.minR2      = 0.60;
cfg.betaTD     = 0.30;

% experiment 
cfg.nIter      = 40;         % REPORT: policy iterations
cfg.nEval      = 300;
cfg.nAccept    = 300;
cfg.stateEp    = 40;
cfg.maxStates  = 1800;
cfg.seed       = 1;
cfg.selSeeds   = [777 778 779];
cfg.repSeeds   = [997 998 999];
end
