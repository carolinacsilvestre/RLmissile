function verify_gradients()
rng(1);
net = mlp_init(6, 32);
net.w = randn(size(net.w))*0.2;     % nonzero weights, or the test is trivial
x = randn(6,1)*0.4;
h = 1e-6;
[~, gw, gx] = mlp_eval(net, x);

ew = 0;
for k = 1:numel(net.w)
    a = net; b = net;
    a.w(k) = a.w(k) + h;  b.w(k) = b.w(k) - h;
    ew = max(ew, abs((mlp_eval(a,x) - mlp_eval(b,x))/(2*h) - gw(k)));
end
ex = 0;
for i = 1:numel(x)
    xa = x; xb = x;
    xa(i) = xa(i) + h;  xb(i) = xb(i) - h;
    ex = max(ex, abs((mlp_eval(net,xa) - mlp_eval(net,xb))/(2*h) - gx(i)));
end

fprintf('dJ/dw  max abs error over %d weights : %.3e\n', numel(net.w), ew);
fprintf('dJ/dx  max abs error over %d inputs  : %.3e\n', numel(x), ex);
if max(ew,ex) < 1e-6
    fprintf('GRADIENTS OK\n');
else
    error('verify_gradients:failed','gradient check failed');
end
end
