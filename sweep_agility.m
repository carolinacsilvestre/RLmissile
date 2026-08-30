function sweep_agility(n, amps, Ts, file)
% Robustness to engagement conditions not seen in training. target agility
% and weave period, swept together. PN is re-tuned over N in every cell; the
% HDP controller is the one trained at 5 g / 3 s and is not retrained.
% cfg.accelNorm and cfg.jerkNorm are literals in cfg_default and do not track
% tgtAMax or weaveT, so no cell rescales the policy's inputs.
if nargin < 1 || isempty(n),    n    = 300;            end
if nargin < 2 || isempty(amps), amps = [2.5 5.0 7.5];  end   % REPORT: [g]
if nargin < 3 || isempty(Ts),   Ts   = [2 3 4 6];      end   % REPORT: [s]
if nargin < 4 || isempty(file), file = 'result_main.mat'; end
G = 9.81;
base  = cfg_default();
seeds = base.repSeeds;
Ns    = 2:0.5:5;

S     = load(file);
netRL = S.best.net;
cfgRL = S.cfg;
net0  = mlp_init(base.nx, base.nHidden);   % zero output layer => exactly PN

nA = numel(amps); nT = numel(Ts);
hitPN = zeros(nT,nA); medPN = hitPN; bestN = hitPN;
hitRL = zeros(nT,nA); medRL = hitRL;

fprintf('\n=== agility x weave period, %d engagements per cell ===\n', n);
fprintf('  a_max [g]  T [s]   PN [%%]  bestN   HDP [%%]   delta\n');
for j = 1:nA
    for i = 1:nT
        c = base; c.tgtAMax = amps(j)*G; c.weaveT = Ts(i);
        h = -inf;
        for k = 1:numel(Ns)
            cc = c; cc.N = Ns(k);
            s = evaluate(cc, net0, n, seeds);
            if s.hit > h
                h = s.hit; medPN(i,j) = s.med; bestN(i,j) = Ns(k);
            end
        end
        hitPN(i,j) = h;

        cR = cfgRL; cR.tgtAMax = amps(j)*G; cR.weaveT = Ts(i);
        s = evaluate(cR, netRL, n, seeds);
        hitRL(i,j) = s.hit; medRL(i,j) = s.med;

        fprintf('   %5.2f     %4.1f   %6.2f    %.1f    %6.2f   %+6.2f\n', ...
                amps(j), Ts(i), hitPN(i,j), bestN(i,j), hitRL(i,j), ...
                hitRL(i,j)-hitPN(i,j));
    end
end

save('sweep_agility.mat','amps','Ts','hitPN','medPN','bestN','hitRL', ...
     'medRL','n','seeds','Ns','file');

emit_latex(amps, Ts, hitPN, bestN, hitRL, n);

figure('Name','agility robustness');
plot(amps, hitPN(Ts==3,:), '-o', amps, hitRL(Ts==3,:), '-s', 'LineWidth',1.5);
grid on; xlabel('target amplitude a_{t,max} [g]'); ylabel('hit rate [%]');
legend('best-N PN','residual HDP','Location','southwest');
title('degradation with agility at T = 3 s');
end


