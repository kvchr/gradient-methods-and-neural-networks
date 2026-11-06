% Part 1, Problem 2: gradient descent and the momentum method on f(x,y) = A x^2 + y^2 / A
% The Hessian is H = diag(2A, 2/A), so its condition number is kappa = max(A^2, 1/A^2).

clc; clearvars; close all;
figdir = fullfile(fileparts(mfilename('fullpath')), '..', 'figures');

As = [0.1 0.5 1 1.2*(1:10)];   % values of A (A and 1/A give the same kappa)
initial = [50; -100];          % starting point
alpha = 0.05;                  % step size (both methods)
beta = 0.9;                    % momentum coefficient
tol = 1e-6;                    % stop when the step is shorter than tol
max_iters = 20000;             % high enough that every run converges

rows = zeros(numel(As), 6);
paths = struct();
for n = 1:numel(As)
    A = As(n);
    grad = @(p) [2*A*p(1); 2/A*p(2)];
    [path_gd, it_gd] = gradient_descent(grad, initial, alpha, tol, max_iters);
    [path_mom, it_mom] = momentum_method(grad, initial, alpha, beta, tol, max_iters);
    assert(it_gd < max_iters && it_mom < max_iters)      % both methods converged

    % Convergence factor per iteration (spectral radius of the linear iteration)
    lambdas = [2*A, 2/A];                                % eigenvalues of H
    rho_gd = max(abs(1 - alpha*lambdas));
    rho_mom = 0;
    for lam = lambdas                                    % z^2 - (1 + beta - alpha*lam) z + beta = 0
        rho_mom = max(rho_mom, max(abs(roots([1, -(1 + beta - alpha*lam), beta]))));
    end
    rows(n, :) = [A, max(A^2, 1/A^2), rho_gd, it_gd, rho_mom, it_mom];

    if any(abs(A - [1.2 6 12]) < 1e-9)
        paths.(sprintf('A%d', round(10*A))) = {path_gd, path_mom};
    end
end
results = array2table(rows, 'VariableNames', {'A', 'kappa', 'rho_GD', 'iter_GD', 'rho_momentum', 'iter_momentum'});
fprintf('alpha = %.2f, beta = %.1f, start (%g, %g), tolerance %g\n', alpha, beta, initial(1), initial(2), tol);
disp(results);
fprintf('Gradient descent is stable only if alpha * max(2A, 2/A) < 2, i.e. %.2f < A < %.0f for alpha = %.2f\n', ...
    alpha, 1/alpha, alpha);

%% Figures
fig = figure('Color', 'w', 'Position', [100 100 620 420]);
up = results.A >= 1; lo = ~up;
cols = lines(2);
plot(results.kappa(up), results.iter_GD(up), '-o', 'Color', cols(1,:), 'LineWidth', 1.3, 'DisplayName', 'Gradient descent, A \geq 1'); hold on;
plot(results.kappa(up), results.iter_momentum(up), '-s', 'Color', cols(2,:), 'LineWidth', 1.3, 'DisplayName', 'Momentum (\beta = 0.9), A \geq 1');
plot(results.kappa(lo), results.iter_GD(lo), 'o', 'Color', cols(1,:), 'MarkerFaceColor', cols(1,:), 'DisplayName', 'Gradient descent, A < 1');
plot(results.kappa(lo), results.iter_momentum(lo), 's', 'Color', cols(2,:), 'MarkerFaceColor', cols(2,:), 'DisplayName', 'Momentum, A < 1');
set(gca, 'XScale', 'log'); grid on;
xlabel('condition number \kappa of the Hessian'); ylabel('iterations');
title(sprintf('Iterations until the step is shorter than 10^{-6} (\\alpha = %.2f)', alpha));
legend('Location', 'northwest');
exportgraphics(fig, fullfile(figdir, 'p1_2_iterations_vs_kappa.png'), 'Resolution', 200);

fig = figure('Color', 'w', 'Position', [100 100 1200 420]);
keys = {'A12', 'A60', 'A120'}; Avals = [1.2 6 12];
[X, Y] = meshgrid(-60:1:60, -110:1:60);
for k = 1:3
    subplot(1, 3, k);
    A = Avals(k);
    contour(X, Y, A*X.^2 + Y.^2/A, 25, 'LineColor', [0.75 0.75 0.75], 'HandleVisibility', 'off'); hold on;
    p = paths.(keys{k});
    plot(p{1}(:,1), p{1}(:,2), 'r.-', 'MarkerSize', 6, 'DisplayName', sprintf('Gradient descent (%d)', size(p{1},1) - 1));
    plot(p{2}(:,1), p{2}(:,2), 'b.-', 'MarkerSize', 6, 'DisplayName', sprintf('Momentum (%d)', size(p{2},1) - 1));
    axis equal; xlim([-60 60]); ylim([-110 60]);
    xlabel('x'); ylabel('y'); title(sprintf('A = %.1f, \\kappa = %.1f', A, max(A^2, 1/A^2)));
    legend('Location', 'northeast');
end
exportgraphics(fig, fullfile(figdir, 'p1_2_paths.png'), 'Resolution', 200);

function [path, iters] = gradient_descent(grad, p, alpha, tol, max_iters)
    % x_{k+1} = x_k - alpha * grad f(x_k)
    path = p';
    for iters = 1:max_iters
        p_next = p - alpha * grad(p);
        path(end+1, :) = p_next'; %#ok<AGROW>
        if norm(p_next - p) < tol
            break
        end
        p = p_next;
    end
end

function [path, iters] = momentum_method(grad, p, alpha, beta, tol, max_iters)
    % v_{k+1} = beta * v_k - alpha * grad f(x_k),  x_{k+1} = x_k + v_{k+1}
    v = zeros(size(p));
    path = p';
    for iters = 1:max_iters
        v = beta * v - alpha * grad(p);
        p_next = p + v;
        path(end+1, :) = p_next'; %#ok<AGROW>
        if norm(p_next - p) < tol
            break
        end
        p = p_next;
    end
end
