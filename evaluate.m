function st = evaluate(cfg, net, nTotal, seeds)
old = rng;
nEach = ceil(nTotal/numel(seeds));
m = zeros(nEach*numel(seeds),1); h = false(size(m)); g = zeros(size(m));
res2 = 0; nres = 0; q = 0;

for s = 1:numel(seeds)
    rng(seeds(s));
    for i = 1:nEach
        q = q+1;
        ep = run_episode(cfg, net);
        m(q) = ep.miss; h(q) = ep.hit; g(q) = ep.G;
        res2 = res2 + sum(ep.log.res.^2); nres = nres + ep.n;
    end
end
rng(old);

m = m(1:nTotal); h = h(1:nTotal); g = g(1:nTotal);
p = mean(h);
st = struct('hit',100*p, 'hitSE',100*sqrt(p*(1-p)/nTotal), 'med',median(m), ...
            'J',mean(g), 'miss',m, 'hits',h, 'returns',g, ...
            'resRms',sqrt(res2/max(nres,1)));
end
