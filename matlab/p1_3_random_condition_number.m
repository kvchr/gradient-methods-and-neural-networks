% Part 1, Problem 3: condition number of Q = A*A' for a random n x n matrix A
% The singular values of Q are the squares of those of A, so cond(Q) = cond(A)^2.

clc; clearvars; close all;
figdir = fullfile(fileparts(mfilename('fullpath')), '..', 'figures');
rng(0);

dims = 2:100;              % dimensions n
num_matrices = 200;        % random matrices per dimension
K = zeros(numel(dims), num_matrices);
logK_A = zeros(numel(dims), num_matrices);
for i = 1:numel(dims)
    n = dims(i);
    for j = 1:num_matrices
        A = randn(n, n);                 % entries independent N(0, 1)
        Q = A * A';                      % symmetric positive definite (almost surely)
        K(i, j) = cond(Q);
        logK_A(i, j) = log(cond(A));
    end
end

med = median(K, 2);
q25 = prctile(K, 25, 2);
q75 = prctile(K, 75, 2);
avg = mean(K, 2);
fprintf('cond(Q) = cond(A)^2 check (n = 100, last matrix): %.3e vs %.3e\n', cond(Q), cond(A)^2);
fprintf('%5s %14s %14s %14s %12s\n', 'n', 'median', 'mean', 'max', 'median/n^2');
for n = [2 5 10 20 50 100]
    i = find(dims == n);
    fprintf('%5d %14.4g %14.4g %14.4g %12.2f\n', n, med(i), avg(i), max(K(i, :)), med(i) / n^2);
end
big = dims >= 50;
cst = mean(med(big) ./ dims(big)'.^2);
fprintf('mean of median/n^2 for n >= 50: %.2f\n', cst);
fprintf('mean of log(cond(A)) - log(n) for n >= 50: %.3f\n', mean(mean(logK_A(big, :), 2) - log(dims(big))'));

%% Figure
fig = figure('Color', 'w', 'Position', [100 100 1150 440]);
subplot(1, 2, 1);
fill([dims fliplr(dims)], [q25' fliplr(q75')], [0.8 0.88 1], 'EdgeColor', 'none', 'DisplayName', '25%-75% of the matrices'); hold on;
loglog(dims, med, 'b-', 'LineWidth', 1.6, 'DisplayName', 'median');
loglog(dims, avg, 'r.', 'MarkerSize', 9, 'DisplayName', 'mean');
loglog(dims, cst * dims.^2, 'k--', 'LineWidth', 1.1, 'DisplayName', sprintf('%.1f n^2', cst));
set(gca, 'XScale', 'log', 'YScale', 'log'); grid on;
xlabel('dimension n'); ylabel('\kappa(Q)');
title(sprintf('\\kappa(AA^T), %d random matrices per n', num_matrices)); legend('Location', 'northwest');
subplot(1, 2, 2);
i = find(dims == 50);
histogram(log10(K(i, :)), 30, 'FaceColor', [0.3 0.5 0.8]);
xlabel('log_{10} \kappa(Q)'); ylabel('number of matrices'); grid on;
title(sprintf('n = 50: median %.3g, mean %.3g, max %.3g', med(i), avg(i), max(K(i, :))));
exportgraphics(fig, fullfile(figdir, 'p1_3_condition_number.png'), 'Resolution', 200);
