function sweep_learning()
% sensitivity to the learning parameters 
% each sweep retrains from scratch. reduce nIter/nEval for a quick pass.
base = cfg_default();
base.nIter = 40;

trust = [0.025 0.05 0.10 0.20 0.40]*9.81;   % REPORT: headline sweep
hitT = zeros(size(trust));
for i = 1:numel(trust)
    c = base; c.trustRms = trust(i);
    fprintf('\n--- trustRms = %.3f g ---\n', trust(i)/9.81);
    r = train(c);
    hitT(i) = r.bestStats.hit;
end

H = [1 3 6 12];                              % REPORT: H=1 is the lecture gradient
hitH = zeros(size(H));
for i = 1:numel(H)
    c = base; c.H = H(i);
    fprintf('\n--- H = %d ---\n', H(i));
    r = train(c);
    hitH(i) = r.bestStats.hit;
end

save('sweep_learning.mat','trust','hitT','H','hitH','base');

figure('Name','learning-parameter sensitivity');
subplot(1,2,1);
semilogx(trust/9.81, hitT, '-o', 'LineWidth',1.5); grid on
xlabel('trust radius [g per iteration]'); ylabel('hit rate [%]');
subplot(1,2,2);
plot(H, hitH, '-o', 'LineWidth',1.5); grid on
xlabel('gradient horizon H'); ylabel('hit rate [%]');
end
