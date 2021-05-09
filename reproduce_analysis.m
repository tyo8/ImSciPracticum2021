% Script that re-runs all data analysis

function [] = reproduce_analysis()
addpath(genpath('dependencies'))
load('reproducing_vars.mat','X','Y','IDP_sig_vec','U','S','V')

UMAP_savename = ['SVM_IDP_UMAP_validation_',...
    datestr(datetime,'mmmdd') '.mat'];

other_savename = ['SVM_IDP_nonUMAP_validation_',...
    datestr(datetime,'mmmdd') '.mat'];

X_sig = X(:,IDP_sig_vec);
Mdl_sig = fitcsvm(X_sig,Y,'KernelFunction','linear',...
'BoxConstraint',1e-3,'KFold',5,'Standardize',false);
kf_loss_sig = kfoldLoss(Mdl_sig);

X_pca_basis = proj_data_svd(X,U,S,V,'prop_var_exp_thresh',1);
pca_kf_loss = svm_vs_pcacomp(X_pca_basis,Y);

save(other_savename,'Mdl_sig','kf_loss_sig','pca_kf_loss')

IDP_UMAP_parsearch(X,Y,'save_name',UMAP_savename)
end