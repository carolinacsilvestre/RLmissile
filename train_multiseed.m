function train_multiseed(seeds, nIter)
% Repeat training over independent seeds and report mean +/- sd.
% Held-out evaluation episodes are identical for every seed, so the spread
% below is training variability, not evaluation noise.
if nargin < 1 || isempty(seeds), seeds = 1:5; end
base = cfg_default();
if nargin > 1 && ~isempty(nIter), base.nIter = nIter; end

n = numel(seeds);
hit = nan(n,1); med = nan(n,1); J = nan(n,1); pn = nan(n,1); C = cell(n,1);

for k = 1:n
    c = base; c.seed = seeds(k);
    fprintf('\n=============== SEED %d  (%d of %d) ===============\n', seeds(k), k, n);
    t0 = tic;
    try
        r = train(c);
        hit(k) = r.bestStats.hit; med(k) = r.bestStats.med;
        J(k)   = r.bestStats.J;   pn(k)  = r.pn.hit;   C{k} = r.curve;
    catch err
        fprintf('  SEED %d FAILED: %s\n', seeds(k), err.message);
    end
    fprintf('  seed %d finished in %.0f s\n', seeds(k), toc(t0));
    save('multiseed.mat','seeds','hit','med','J','pn','C','base');
end

ok = isfinite(hit);
if ~any(ok), error('train_multiseed:allFailed','no seed completed'); end

fprintf('\n=========== MULTI-SEED SUMMARY (%d of %d) ===========\n', sum(ok), n);
fprintf('  PN baseline    %6.2f %%\n', mean(pn(ok)));
fprintf('  HDP hit rate   %6.2f +/- %.2f %%   [%.2f .. %.2f]\n', ...
        mean(hit(ok)), std(hit(ok)), min(hit(ok)), max(hit(ok)));
fprintf('  HDP median     %6.2f +/- %.2f m\n', mean(med(ok)), std(med(ok)));
fprintf('  HDP return J   %6.3f +/- %.3f\n', mean(J(ok)), std(J(ok)));

figure('Name','multi-seed summary');
bar(1:n, hit, 0.6); hold on
plot([0 n+1], [1 1]*mean(pn(ok)), 'k--', 'LineWidth',1.2);
plot([0 n+1], [1 1]*mean(hit(ok)), 'r-', 'LineWidth',1.2);
grid on; xlim([0 n+1]);
set(gca,'XTick',1:n,'XTickLabel',arrayfun(@(v) sprintf('%d',v), seeds, 'UniformOutput', false));
xlabel('training seed'); ylabel('best held-out hit rate [%]');
legend('per seed','PN baseline','mean over seeds','Location','southeast');
title(sprintf('%.2f \\pm %.2f %% over %d seeds', mean(hit(ok)), std(hit(ok)), sum(ok)));

L = max(cellfun(@(x) size(x,1), C(ok)));
M = nan(L, sum(ok)); j = 0;
for k = 1:n
    if ok(k), j = j+1; M(1:size(C{k},1),j) = C{k}(:,5); end
end
it = (0:L-1).';
mu = nan(L,1); sd = nan(L,1);
for i = 1:L
    v = M(i, isfinite(M(i,:)));
    mu(i) = mean(v); sd(i) = std(v);
end
if any(cellfun(@(x) size(x,1), C(ok)) < L)
    fprintf('  note: seeds ran unequal lengths; band averages over available seeds\n');
end

figure('Name','multi-seed learning curve');
fill([it; flipud(it)], [mu-sd; flipud(mu+sd)], [0.80 0.86 0.95], 'EdgeColor','none'); hold on
plot(it, mu, '-o', 'LineWidth',1.4);
plot([min(it) max(it)], [1 1]*mean(pn(ok)), 'k--', 'LineWidth',1.2);
grid on; xlabel('policy iteration'); ylabel('hit rate [%]');
legend('\pm1 sd across seeds','mean','PN baseline','Location','southeast');
end
