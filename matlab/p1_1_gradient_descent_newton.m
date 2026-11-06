% Part 1, Problem 1: gradient descent with several step sizes, compared with Newton's method
% f(x,y) = x^2 + (y-1)^2 + (x-y)^4, minimum at (0.25, 0.75) with f = 0.1875.
% The gradient-descent loop follows the structure of the tutorial
% "2022 How to do Gradient Descent Optimization Algorithm in MATLAB | MATLAB Tutorial".

clc; clearvars; close all;
figdir = fullfile(fileparts(mfilename('fullpath')), '..', 'figures');

f = @(x,y) x.^2 + (y-1).^2 + (x-y).^4;
syms xs ys
fs = xs.^2 + (ys-1).^2 + (xs-ys).^4;

% Gradient and Hessian (symbolic, then turned into MATLAB functions)
Df_x = diff(fs, xs);
Df_y = diff(fs, ys);
grad_x = matlabFunction(Df_x, 'Vars', [xs ys]);
grad_y = matlabFunction(Df_y, 'Vars', [xs ys]);
Hessian = matlabFunction([diff(Df_x, xs), diff(Df_x, ys); diff(Df_y, xs), diff(Df_y, ys)], 'Vars', [xs ys]);

x_star = [0.25 0.75];                 % solves grad f = 0 (x + y = 1 and 2x + 4(2x-1)^3 = 0)
f_star = f(x_star(1), x_star(2));
start = [1 1];                        % initial point
tol = 1e-6;                           % stop when the step is shorter than tol
max_iters = 5000;

%% Gradient descent for several step sizes
alphas = [0.01 0.05 0.1 0.2 0.3 0.4];
paths = cell(size(alphas));
fprintf('Gradient descent from (1, 1), tolerance %g:\n', tol);
for j = 1:numel(alphas)
    alpha = alphas(j);
    pts = start;
    status = 'no convergence';
    for i = 1:max_iters
        % next point: subtract the step size times the gradient
        p = pts(end, :);
        p_next = p - alpha * [grad_x(p(1), p(2)), grad_y(p(1), p(2))];
        pts(end+1, :) = p_next; %#ok<AGROW>
        if any(~isfinite(p_next)) || norm(p_next) > 1e6
            status = 'diverges';
            break
        end
        if norm(p_next - p) < tol
            status = 'converges';
            break
        end
    end
    paths{j} = pts;
    last = pts(end, :);
    switch status
        case 'converges'
            fprintf('  alpha = %.2f: converges after %d iterations, last point (%.4f, %.4f), f = %.6g\n', ...
                alpha, size(pts, 1) - 1, last(1), last(2), f(last(1), last(2)));
        case 'diverges'
            fprintf('  alpha = %.2f: diverges, |x| > 1e6 after %d iterations\n', alpha, size(pts, 1) - 1);
        otherwise
            prev = pts(end-1, :);
            fprintf('  alpha = %.2f: no convergence in %d iterations, it alternates between (%.4f, %.4f) and (%.4f, %.4f)\n', ...
                alpha, size(pts, 1) - 1, prev(1), prev(2), last(1), last(2));
    end
end
% Near the minimum gradient descent behaves like x_{k+1} - x* = (I - alpha*H)(x_k - x*), H the Hessian at x*
lam = eig(Hessian(x_star(1), x_star(2)));
fprintf('Hessian at the minimum: eigenvalues %g and %g, so gradient descent is stable near x* only for alpha < 2/%g = %.2f\n', ...
    lam(1), lam(2), lam(2), 2 / lam(2));
for alpha = alphas(1:4)
    fprintf('  alpha = %.2f: error shrinks by a factor max|1 - alpha*lambda| = %.2f per iteration\n', alpha, max(abs(1 - alpha * lam)));
end

%% Newton's method: x_{k+1} = x_k - H^{-1} grad f
pts_newton = start;
for i = 1:max_iters
    p = pts_newton(end, :);
    delta = Hessian(p(1), p(2)) \ [grad_x(p(1), p(2)); grad_y(p(1), p(2))];
    pts_newton(end+1, :) = p - delta'; %#ok<AGROW>
    if norm(delta) < tol
        break
    end
end
last = pts_newton(end, :);
fprintf('Newton: converges after %d iterations, last point (%.4f, %.4f), f = %.6g\n', ...
    size(pts_newton, 1) - 1, last(1), last(2), f(last(1), last(2)));

%% Figures
[X, Y] = meshgrid(-0.5:0.02:1.5, -0.5:0.02:1.5);
fig = figure('Color', 'w', 'Position', [100 100 1150 460]);
subplot(1, 2, 1);
contour(X, Y, f(X, Y), logspace(-0.8, 1, 18), 'LineColor', [0.7 0.7 0.7], 'HandleVisibility', 'off'); hold on;
cols = lines(numel(alphas));
for j = [1 4 5]                       % alpha = 0.01, 0.2 and 0.3
    pts = paths{j};
    plot(pts(:,1), pts(:,2), '.-', 'Color', cols(j,:), 'MarkerSize', 8, ...
        'DisplayName', sprintf('GD, \\alpha = %.2f', alphas(j)));
end
plot(pts_newton(:,1), pts_newton(:,2), 'ko-', 'LineWidth', 1.2, 'DisplayName', 'Newton');
plot(x_star(1), x_star(2), 'kp', 'MarkerSize', 12, 'MarkerFaceColor', 'k', 'DisplayName', 'minimum (0.25, 0.75)');
axis equal; xlim([-0.5 1.5]); ylim([-0.5 1.5]);
xlabel('x'); ylabel('y'); title('Paths from (1, 1)'); legend('Location', 'southwest');
subplot(1, 2, 2);
for j = 1:4                           % the step sizes that converge
    pts = paths{j};
    semilogy(0:size(pts,1)-1, max(f(pts(:,1), pts(:,2)) - f_star, 1e-16), 'Color', cols(j,:), 'LineWidth', 1.3, ...
        'DisplayName', sprintf('GD, \\alpha = %.2f', alphas(j))); hold on;
end
semilogy(0:size(pts_newton,1)-1, max(f(pts_newton(:,1), pts_newton(:,2)) - f_star, 1e-16), 'ko-', 'LineWidth', 1.3, 'DisplayName', 'Newton');
xlim([0 160]); ylim([1e-14 10]); grid on;
xlabel('iteration k'); ylabel('f(x_k) - f^*'); title('Distance from the minimum value'); legend;
exportgraphics(fig, fullfile(figdir, 'p1_1_gradient_descent_newton.png'), 'Resolution', 200);
