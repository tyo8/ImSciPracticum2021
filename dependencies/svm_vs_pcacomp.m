% Note: assumes X is in PCA basis and columns are sorted by decreasing
% variance explained.

function [svm_acc] = svm_vs_pcacomp(X,Y)

n = round(size(X,2)/5);

subsamp = linspace(0,log10(size(X,2)),n);

n_dims = unique(round(10.^subsamp));

svm_acc = zeros(size(n_dims));

for i=1:length(n_dims)
    Mdl = fitcsvm(X(:,1:n_dims(i)),Y,'KernelFunction','linear',...
        'BoxConstraint',1e-3,'KFold',5,'Standardize',false);
    
    svm_acc(i)=kfoldLoss(Mdl);
end

end
