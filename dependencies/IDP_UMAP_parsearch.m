function [kf_loss,Model_set,X_UMAP] = IDP_UMAP_parsearch(X,Y,varargin)

def_savename = ['IDP_UMAP_parsearch_SVM_',...
    datestr(datetime,'mmmdd') '.mat'];

def_dimlist = [3 17 35 69 100];
def_mindist_list = (0.1:0.05:0.5);

p = inputParser;
addParameter(p,'save_flag',true)
addParameter(p,'save_name',def_savename)
addParameter(p,'dimlist',def_dimlist)
addParameter(p,'mindist_list',def_mindist_list)
addParameter(p,'n_epochs',300)
parse(p,varargin{:})

save_flag = p.Results.save_flag;
save_name = p.Results.save_name;
dimlist = p.Results.dimlist;
mindist_list = p.Results.mindist_list;
n_epochs = p.Results.n_epochs;

m = length(dimlist);
n = length(mindist_list);

kf_loss = zeros(m,n);
X_UMAP = cell(m,n);
Model_set = cell(m,n);

UMAP_pars.metric = 'correlation';
UMAP_pars.IncludeTies = true;
UMAP_pars.n_components = dimlist;
UMAP_pars.min_dist = mindist_list;

for i=1:m
    for j=1:n
        X_red_ij = doIDP_umap(X,'target_dim',dimlist(i),'n_epochs',...
            n_epochs,'min_dist',mindist_list(j),'save_flag',false);
        
        Mdl_ij = fitcsvm(X_red_ij,Y,'KernelFunction','gaussian',...
        'BoxConstraint',1e-3,'KFold',5,'Standardize',false);
    
        X_UMAP{i,j} = X_red_ij;
        Model_set{i,j} = Mdl_ij;
        
        kf_loss(i,j)=kfoldLoss(Mdl_ij);
    end
end

if save_flag
    save(save_name,'kf_loss','X_UMAP','Model_set',...
        'UMAP_pars','dimlist','mindist_list')
end
end