vars = load('training_data.mat','X','Y');

hyperpars = {{'BoxConstraint'}};
hp_name = {'boxcon_pars.mat'};

svm_hyperpar_optim_tsts(vars.X,vars.Y,'hyperpars',hyperpars,...
    'hp_name',hp_name,'n_evals',200);