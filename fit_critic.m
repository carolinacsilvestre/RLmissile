function [critic, st] = fit_critic(cfg, net, critic, nTrain, nVal, nEpoch, seed)
[Xt,Gt] = collect(cfg, net, nTrain, seed);
[Xv,Gv] = collect(cfg, net, nVal,   seed+99);

if isempty(critic)
    critic = mlp_init(cfg.nx, cfg.nHidden);
    critic.w(end) = mean(Gt);
end

if strcmpi(cfg.criticMethod,'td')
    critic = fit_td(cfg, net, critic, nTrain, seed);
    Pv = predict(critic, Xv);
    st = struct('R2', 1-sum((Pv-Gv).^2)/max(sum((Gv-mean(Gv)).^2),eps), ...
                'trainR2',NaN, 'history',[], 'valTarget',Gv, 'valPrediction',Pv);
    st.passed = st.R2 >= cfg.minR2;
    return
end

w = critic.w; m = zeros(size(w)); v = m; t = 0;
best = -inf; bestW = w; stale = 0; hist = nan(nEpoch,2);
old = rng; rng(seed+50000);

for e = 1:nEpoch
    ord = randperm(size(Xt,2));
    for j = 1:cfg.batchSize:numel(ord)
        ix = ord(j:min(j+cfg.batchSize-1, numel(ord)));
        critic.w = w;
        g = loss_grad(critic, Xt(:,ix), Gt(ix)) + cfg.weightDecay*w;
        ng = norm(g); if ng > cfg.fitClip, g = g*(cfg.fitClip/ng); end
        t = t+1;
        m = 0.9*m + 0.1*g;
        v = 0.999*v + 0.001*(g.^2);
        w = w - cfg.learnRate*(m/(1-0.9^t))./(sqrt(v/(1-0.999^t))+1e-8);
    end
    critic.w = w;
    Pv = predict(critic, Xv);
    r2 = 1 - sum((Pv-Gv).^2)/max(sum((Gv-mean(Gv)).^2), eps);
    hist(e,:) = [e r2];
    if r2 > best+1e-5, best = r2; bestW = w; stale = 0; else, stale = stale+1; end
    if stale >= cfg.patience, break, end
end
rng(old);

critic.w = bestW;
Pt = predict(critic, Xt); Pv = predict(critic, Xv);
st = struct('R2', 1-sum((Pv-Gv).^2)/max(sum((Gv-mean(Gv)).^2),eps), ...
            'trainR2', 1-sum((Pt-Gt).^2)/max(sum((Gt-mean(Gt)).^2),eps), ...
            'history', hist(1:find(isfinite(hist(:,1)),1,'last'),:), ...
            'valTarget', Gv, 'valPrediction', Pv);
st.passed = st.R2 >= cfg.minR2;
end

% complete discounted returns under the frozen policy 
function [X,G] = collect(cfg, net, nEp, seed)
old = rng; rng(seed);
cap = nEp*ceil(cfg.maxSteps/cfg.stride);
X = zeros(cfg.nx, cap); G = zeros(1, cap); q = 0;
for ep = 1:nEp
    [env, obs] = env_reset(cfg);
    Xi = zeros(cfg.nx, cfg.maxSteps); R = zeros(1, cfg.maxSteps);
    k = 0; done = false;
    while ~done && k < cfg.maxSteps
        k = k+1; Xi(:,k) = obs.x;
        u = policy_action(net, obs, cfg);
        [env, obs, done] = env_step(env, u, cfg);
        R(k) = reward(obs, u, done, env.miss, cfg);
    end
    Gi = zeros(1,k); z = 0;
    for i = k:-1:1, z = R(i) + cfg.gamma*z; Gi(i) = z; end
    take = 1:cfg.stride:k; nT = numel(take);
    X(:,q+1:q+nT) = Xi(:,take); G(q+1:q+nT) = Gi(take); q = q+nT;
end
X = X(:,1:q); G = G(1:q); rng(old);
end

% TD(0) alternative, lecture eq. (9)-(10) 
function critic = fit_td(cfg, net, critic, nEp, seed)
old = rng; rng(seed);
for ep = 1:nEp
    [env, obs] = env_reset(cfg);
    k = 0; done = false;
    while ~done && k < cfg.maxSteps
        k = k+1;
        x = obs.x;
        u = policy_action(net, obs, cfg);
        [env, obs2, done] = env_step(env, u, cfg);
        r = reward(obs2, u, done, env.miss, cfg);
        [J, gw] = mlp_eval(critic, x);
        if done, target = r; else, target = r + cfg.gamma*mlp_eval(critic, obs2.x); end
        critic.w = critic.w + (cfg.betaTD/max(gw.'*gw,eps))*(target-J)*gw;
        obs = obs2;
    end
end
rng(old);
end

function Y = predict(net, X)
nx = net.nx; nh = net.nh;
W1 = reshape(net.w(1:nh*nx), nh, nx);
b1 = net.w(nh*nx+1 : nh*nx+nh);
W2 = reshape(net.w(nh*nx+nh+1 : nh*nx+2*nh), 1, nh);
Y  = W2*tanh(W1*X + b1) + net.w(end);
end

function gw = loss_grad(net, X, T)
nx = net.nx; nh = net.nh;
W1 = reshape(net.w(1:nh*nx), nh, nx);
b1 = net.w(nh*nx+1 : nh*nx+nh);
W2 = reshape(net.w(nh*nx+nh+1 : nh*nx+2*nh), 1, nh);
H  = tanh(W1*X + b1);
E  = (W2*H + net.w(end)) - reshape(T,1,[]);
dY = E/max(size(X,2),1);
dZ = (W2.'*dY).*(1-H.^2);
gW1 = dZ*X.';
gw  = [gW1(:); sum(dZ,2); (dY*H.').'; sum(dY,2)];
end
