function sweep_envelope(n)
% pure eval
if nargin < 1 || isempty(n), n = 300; end
base = cfg_default();
seeds = base.repSeeds;
Ns = 2:0.5:5;

% autopilot lag, weaving target 
taus = [0 0.1 0.2 0.3 0.4 0.5];
hitA = zeros(size(taus)); medA = hitA; NA = hitA;
fprintf('\n=== A. best-N PN vs autopilot lag (weaving target) ===\n');
fprintf('  tau [s]   hit [%%]   median [m]   best N\n');
for i = 1:numel(taus)
    c = base; c.tau = taus(i);
    [hitA(i), medA(i), NA(i)] = bestPN(c, Ns, n, seeds);
    fprintf('  %5.2f    %6.2f    %8.2f     %.1f\n', taus(i), hitA(i), medA(i), NA(i));
end

% launch heading error, straight target 

hes = [5 10 15 20 25];
hitB = zeros(size(hes)); medB = hitB; NB = hitB;
fprintf('\n=== B. best-N PN vs launch heading error (straight target) ===\n');
fprintf('  he [deg]  hit [%%]   median [m]   best N\n');
for i = 1:numel(hes)
    c = base; c.tgtAMax = 0; c.he0 = [-hes(i) hes(i)]*pi/180;
    [hitB(i), medB(i), NB(i)] = bestPN(c, Ns, n, seeds);
    fprintf('  %5.0f    %6.2f    %8.2f     %.1f\n', hes(i), hitB(i), medB(i), NB(i));
end

% weave period, PN and (if available) the trained controller 
Ts = [1 1.5 2 2.5 3 4 5 6 8 10];
hitC = zeros(size(Ts)); medC = hitC; NC = hitC;
hitR = nan(size(Ts)); medR = hitR;
haveRL = exist('result_main.mat','file');
if haveRL, S = load('result_main.mat'); end
fprintf('\n=== C. hit rate vs weave period ===\n');
if haveRL
    fprintf('  T [s]   PN [%%]  PN med   bestN   HDP [%%]  HDP med\n');
else
    fprintf('  T [s]   PN [%%]  PN med   bestN\n');
end
for i = 1:numel(Ts)
    c = base; c.weaveT = Ts(i);
    [hitC(i), medC(i), NC(i)] = bestPN(c, Ns, n, seeds);
    if haveRL
        cR = S.cfg; cR.weaveT = Ts(i);       % normalisers stay frozen
        s = evaluate(cR, S.best.net, n, seeds);
        hitR(i) = s.hit; medR(i) = s.med;
        fprintf('  %5.1f  %6.2f  %6.2f    %.1f   %6.2f  %6.2f\n', ...
                Ts(i), hitC(i), medC(i), NC(i), hitR(i), medR(i));
    else
        fprintf('  %5.1f  %6.2f  %6.2f    %.1f\n', Ts(i), hitC(i), medC(i), NC(i));
    end
end
[~,iw] = min(hitC);
fprintf('  worst weave period for PN in this range: T = %.1f s (%.2f%%)\n', Ts(iw), hitC(iw));
if iw == 1 || iw == numel(Ts)
    fprintf('  NOTE: the minimum sits at the edge of the swept range, so no\n');
    fprintf('        interior worst case was found. Widen Ts or revise the text.\n');
end

save('sweep_envelope.mat','taus','hitA','medA','NA','hes','hitB','medB','NB', ...
     'Ts','hitC','medC','NC','hitR','medR','n','seeds','Ns');

figure('Name','envelope sweeps');
subplot(1,3,1);
plot(taus, hitA, '-o', 'LineWidth',1.5); grid on
xlabel('autopilot lag \tau [s]'); ylabel('best-N PN hit rate [%]');
title('weaving target');
subplot(1,3,2);
plot(hes, hitB, '-o', 'LineWidth',1.5); grid on
xlabel('launch heading error [deg]'); ylabel('best-N PN hit rate [%]');
title('straight target');
subplot(1,3,3);
plot(Ts, hitC, '-o', 'LineWidth',1.5); hold on
if haveRL
    plot(Ts, hitR, '-s', 'LineWidth',1.5);
    legend('PN','residual HDP','Location','best');
end
grid on; xlabel('weave period T [s]'); ylabel('hit rate [%]');
title('frequency response');
end

% best navigation constant at one operating point
function [hit, med, bestN] = bestPN(c, Ns, n, seeds)
net = mlp_init(c.nx, c.nHidden);   % zero output layer => exactly PN
hit = -inf; med = NaN; bestN = Ns(1);
for k = 1:numel(Ns)
    cc = c; cc.N = Ns(k);
    s = evaluate(cc, net, n, seeds);
    if s.hit > hit, hit = s.hit; med = s.med; bestN = Ns(k); end
end
end
