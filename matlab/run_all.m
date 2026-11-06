function run_all()
% Runs every MATLAB script of the assignment, saves the figures in ../figures
% and the printed output of each script in ../outputs/<script>.txt

here = fileparts(mfilename('fullpath'));
outdir = fullfile(here, '..', 'outputs');
if ~exist(outdir, 'dir'), mkdir(outdir); end
if ~exist(fullfile(here, '..', 'figures'), 'dir'), mkdir(fullfile(here, '..', 'figures')); end
scripts = {'p1_1_gradient_descent_newton', 'p1_2_condition_number_momentum', ...
    'p1_3_random_condition_number', 'p1_4_subgradient_piecewise_linear', ...
    'p1_5_stochastic_gradient', 'p2_1_relu_network'};
for k = 1:numel(scripts)
    logfile = fullfile(outdir, [scripts{k} '.txt']);
    if exist(logfile, 'file'), delete(logfile); end
    diary(logfile);
    run_one(fullfile(here, [scripts{k} '.m']));
    diary off;
    close all;
end
end

function run_one(file)
% Each script clears its own workspace, so it runs inside this helper.
run(file);
end
