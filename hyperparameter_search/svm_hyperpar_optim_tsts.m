function [] = svm_hyperpar_optim_tsts(X,Y,varargin)
def_hyperpars = {...
    {'BoxConstraint','Standardize'},...
    {'PolynomialOrder','Standardize'}};
def_hp_names = {...
    'box_con_std.mat',...
    'poly_ord_std.mat'};

p = inputParser;
addParameter(p,'hyperpars',def_hyperpars)
addParameter(p,'hp_names',def_hp_names)
addParameter(p,'n_evals',250)
parse(p,varargin{:})

hyperpars = p.Results.hyperpars;
hp_names = p.Results.hp_names;
n_evals = p.Results.n_evals;

for i=1:length(hyperpars)
    if any(contains(hyperpars{i},'Polynomial'))
        kern_fn = 'polynomial';
    else
        kern_fn = 'gaussian';
    end
    hyperpar_optim(X,Y,hyperpars{i},hp_names{i},...
        'kern_fn',kern_fn,'n_evals',n_evals);
    
    fprintf([newline newline 'Optimization %1d completed.'...
        newline newline newline],i);
end
end

function [] = hyperpar_optim(X,Y,hyperpars,hpar_save_name,varargin)

p = inputParser;
addParameter(p,'kern_fn','gaussian')
addParameter(p,'n_evals',250)
parse(p,varargin{:})

kern_fn = p.Results.kern_fn;
n_evals = p.Results.n_evals;

hp_optim_opts = struct(...
    'MaxObjectiveEvaluations',n_evals,...
    'Verbose',2,'Kfold',5,'Repartition',true,...
    'SaveIntermediateResults',true,...
    'UseParallel',false,'ShowPlots',false,...
    'AcquisitionFunctionName','expected-improvement-plus');

disp(['Searching ' hyperpars{:} ' param space...' newline])

tic
Mdl = fitcsvm(X,Y,'KernelFunction',kern_fn,...
    'Solver','L1QP','Standardize',true,...
    'KernelScale','auto',...
    'OptimizeHyperparameters',hyperpars,...
    'HyperparameterOptimizationOptions',hp_optim_opts);
train_time = toc;

tic
CVMdl = crossval(Mdl);
miscl_rate = kfoldLoss(CVMdl);
CVtime = toc;

fprintf('Done. Elapsed time = %2f sec',CVtime+train_time)
disp(newline)

savedir = datestr(datetime,'mmmdd');
if ~exist(savedir,'dir')
    mkdir(savedir)
end
hpar_save_name = [savedir filesep hpar_save_name];

save(hpar_save_name,'hyperpars','hp_optim_opts',...
    'miscl_rate','Mdl','CVMdl','train_time','CVtime','-v7.3')
end