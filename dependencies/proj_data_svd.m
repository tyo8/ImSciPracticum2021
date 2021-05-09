function [X_red] = proj_data_svd(X,U,S,V,varargin)

assert(size(X,1) == size(U,1),'Number of subjects is not consistent.')

p = inputParser;
addParameter(p,'prop_var_exp_thresh',0.5)
parse(p,varargin{:})

pve_thresh = p.Results.prop_var_exp_thresh;

Svals = diag(S);
K = sum(Svals);
pvar_exp = zeros(size(Svals));

for i=1:length(Svals)
    pvar_exp(i) = sum(Svals(1:i))/K;
end

N_sval = find(pvar_exp >= pve_thresh,1,'first');

Sv_r = Svals(1:N_sval);
Sr = diag(Sv_r);
Sr = [Sr, zeros(N_sval,size(V,1) - N_sval)];

Sr_inv = Sr';
for i=1:N_sval
    Sr_inv(i,i) = 1/Sv_r(i);
end

X_red = X*V*Sr_inv;
end