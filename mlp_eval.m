function [y, gw, gx] = mlp_eval(net, x)
nx = net.nx; nh = net.nh;
W1 = reshape(net.w(1:nh*nx), nh, nx);
b1 = net.w(nh*nx+1 : nh*nx+nh);
W2 = reshape(net.w(nh*nx+nh+1 : nh*nx+2*nh), 1, nh);
b2 = net.w(end);

h = tanh(W1*x + b1);
y = W2*h + b2;

if nargout > 1
    dz  = W2.'.*(1-h.^2);
    gW1 = dz*x.';
    gw  = [gW1(:); dz; h; 1];
    gx  = W1.'*dz;
end
end
