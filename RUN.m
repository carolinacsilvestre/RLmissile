% Entry point. Uncomment one block.
cfg = cfg_default();

% (1) figures from an existing trained run
% make_figures('result_main.mat')

% (2) single training run
% r = train(cfg);  make_figures('result_seed1.mat');

% (3) sensitivity to the learning parameters
% sweep_learning

% (4) multi-seed statistics
% train_multiseed(1:5)

% (5) sensitivity to the objective: discount and reward shape
% sweep_reward

% (6) operating envelope: autopilot lag, heading error, weave period
% sweep_envelope

% (7) critic: Monte Carlo vs TD(0)
% compare_critic

% (8) baselines: PN vs APN
% eval_apn(300)

% (9) verification
% verify_plant
% verify_gradients
