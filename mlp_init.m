function net = mlp_init(nx, nh)
s  = 0.35*sqrt(2/(nx+nh));
W1 = s*randn(nh,nx);
net.nx = nx;
net.nh = nh;
net.w  = [W1(:); zeros(nh,1); zeros(nh,1); 0];
end
