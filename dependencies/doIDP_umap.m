function [reduced_IDP,UMAP_pars] = doIDP_umap(X,varargin)

def_savename = ['IDP_UMAP_results_' datestr(datetime,'mmmdd')];

p=inputParser;
addParameter(p,'save_flag',true)
addParameter(p,'savename',def_savename)
addParameter(p,'distance','correlation')
addParameter(p,'target_dim',round(size(X,2)/50))
addParameter(p,'min_dist',0.3)
addParameter(p,'n_epochs',500)
parse(p,varargin{:})

distance = p.Results.distance;
savename = p.Results.savename;
target_dim = p.Results.target_dim;
min_dist = p.Results.min_dist;
save_flag = p.Results.save_flag;
n_epochs = p.Results.n_epochs;

savename = [savename '_' num2str(target_dim) 'D.mat'];

IDP_parnames = cell(1,size(X,2));
for i=1:length(IDP_parnames)
    IDP_parnames{i} = sprintf('IDP_%1d',i);
end

[reduced_IDP,umap_IDP,cluster_IDs,extras] = run_umap(X,'method','MEX',...
    'metric',distance,'NSMethod','nn_descent','IncludeTies',true,...
    'parameter_names',IDP_parnames,'n_components',...
    target_dim,'min_dist',min_dist,'n_epochs',n_epochs);

UMAP_pars.metric = distance;
UMAP_pars.IncludeTies = true;
UMAP_pars.parameter_names = IDP_parnames;
UMAP_pars.n_components = target_dim;
UMAP_pars.min_dist = min_dist;
UMAP_pars.n_epochs = n_epochs;

clear i p

if save_flag
    save(savename)
end
end