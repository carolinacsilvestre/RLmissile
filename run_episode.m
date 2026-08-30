function ep = run_episode(cfg, net)
[env, obs] = env_reset(cfg);
n = cfg.maxSteps;
L.t=zeros(n,1); L.xm=zeros(n,1); L.ym=zeros(n,1); L.xt=zeros(n,1); L.yt=zeros(n,1);
L.a=zeros(n,1); L.res=zeros(n,1); L.lamdot=zeros(n,1); L.atEst=zeros(n,1); L.r=zeros(n,1);

G = 0; k = 0; done = false;
while ~done && k < n
    k = k+1;
    [u,~,~,uRes] = policy_action(net, obs, cfg);
    L.t(k)=env.t; L.xm(k)=env.xm; L.ym(k)=env.ym; L.xt(k)=env.xt; L.yt(k)=env.yt;
    L.a(k)=u; L.res(k)=uRes; L.lamdot(k)=obs.lamdot; L.atEst(k)=env.atEst;

    [env, obs, done] = env_step(env, u, cfg);
    r = reward(obs, u, done, env.miss, cfg);
    L.r(k) = r;
    G = G + cfg.gamma^(k-1)*r;
end

f = fieldnames(L);
for i = 1:numel(f), L.(f{i}) = L.(f{i})(1:k); end
ep = struct('miss',env.miss, 'hit',env.hit, 'G',G, 'n',k, 'log',L);
end
