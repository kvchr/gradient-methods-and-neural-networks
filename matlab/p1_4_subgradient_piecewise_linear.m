% Part 1, Problem 4: the (sub)gradient method on a piecewise-linear function
% f(x,y) = max(x + y, 0.9x - 1.1y + 1, -0.8x + 1.2y - 1, 2 - 1.1x - 0.9y)
% At each point the gradient of the largest (active) piece is used.

clc; clearvars; close all;
figdir = fullfile(fileparts(mfilename('fullpath')), '..', 'figures');

G = [1 1; 0.9 -1.1; -0.8 1.2; -1.1 -0.9];      % gradients of the four pieces (rows)
c = [0; 1; -1; 2];                             % constant terms
f = @(p) max(G * p(:) + c);

%% True minimum: the corner where pieces 1, 2 and 4 are equal
% Solve  g_i' x + c_i = t  for i = 1, 2, 4  (unknowns x, y, t)
act = [1 2 4];
sol = [G(act, :), -ones(3, 1)] \ (-c(act));
x_star = sol(1:2); f_star = sol(3);
% Optimality certificate: 0 = sum lambda_i g_i with lambda_i >= 0, sum lambda_i = 1
lambda = [G(act, :)'; ones(1, 3)] \ [0; 0; 1];
fprintf('Minimum: x* = (%.4f, %.4f), f* = %.4f\n', x_star, f_star);
fprintf('  value of piece 3 there: %.4f (below f*, so not active)\n', G(3, :) * x_star + c(3));
fprintf('  lambda = (%.4f, %.4f, %.4f), all positive, so 0 is in the subdifferential\n', lambda);
fprintf('  gradient lengths of the pieces: %s (never zero)\n', mat2str(vecnorm(G, 2, 2)', 3));

%% Runs from (0, 0)
start = [0; 0];
iters = 2000;
steps = {0.5, 0.1, 0.01, 'diminishing'};
names = {'\alpha = 0.5', '\alpha = 0.1', '\alpha = 0.01', '\alpha_k = 1/k'};
runs = struct('path', {}, 'fbest', {});
fprintf('\n%d iterations from (0, 0):\n', iters);
for r = 1:numel(steps)
    p = start;
    path = zeros(iters + 1, 2); path(1, :) = p';
    fvals = zeros(iters + 1, 1); fvals(1) = f(p);
    for k = 1:iters
        if ischar(steps{r}), alpha = 1 / k; else, alpha = steps{r}; end
        [~, idx] = max(G * p + c);         % active piece
        p = p - alpha * G(idx, :)';        % step along its negative gradient
        path(k + 1, :) = p';
        fvals(k + 1) = f(p);
    end
    fbest = cummin(fvals);                 % the method is not monotone: keep the best value so far
    runs(r).path = path; runs(r).fbest = fbest;
    tail = path(end-199:end, :);           % the last 200 points
    fprintf('  %-14s best f = %.5f (f - f* = %.1e), last point (%.4f, %.4f), last 200 points within %.4f of x*\n', ...
        strrep(names{r}, '\', ''), fbest(end), fbest(end) - f_star, path(end, 1), path(end, 2), ...
        max(vecnorm(tail' - x_star)));
end
% What the stopping rule "step shorter than 1e-3" would do
fprintf('\nWith the stopping rule |x_{k+1} - x_k| < 1e-3:\n');
fprintf('  fixed steps never stop, because the step is alpha * |g| >= %.3f * alpha\n', min(vecnorm(G, 2, 2)));
fprintf('  alpha_k = 1/k stops as soon as |g|/k < 1e-3, i.e. after about %d iterations, wherever the point is\n', ...
    ceil(min(vecnorm(G, 2, 2)) / 1e-3));

%% Figures
[X, Y] = meshgrid(-0.25:0.01:1.25, -0.25:0.01:1.25);
Z = max(max(X + Y, 0.9*X - 1.1*Y + 1), max(-0.8*X + 1.2*Y - 1, 2 - 1.1*X - 0.9*Y));
fig = figure('Color', 'w', 'Position', [100 100 1150 460]);
subplot(1, 2, 1);
contour(X, Y, Z, 30, 'LineColor', [0.75 0.75 0.75], 'HandleVisibility', 'off'); hold on;
cols = lines(numel(steps));
for r = [2 3 4]
    pth = runs(r).path;
    plot(pth(:, 1), pth(:, 2), '.-', 'Color', cols(r, :), 'MarkerSize', 7, 'DisplayName', names{r});
end
plot(x_star(1), x_star(2), 'kp', 'MarkerSize', 13, 'MarkerFaceColor', 'k', 'DisplayName', 'minimum');
axis equal; xlim([-0.25 1.25]); ylim([-0.25 1.25]);
xlabel('x'); ylabel('y'); title('Paths from (0, 0) (grey: level lines of f)'); legend('Location', 'northeast');
subplot(1, 2, 2);
for r = 1:numel(steps)
    loglog(1:iters + 1, max(runs(r).fbest - f_star, 1e-16), 'LineWidth', 1.4, 'Color', cols(r, :), 'DisplayName', names{r}); hold on;
end
grid on; xlabel('iteration k'); ylabel('f_{best}(k) - f^*'); xlim([1 iters + 1]); ylim([1e-5 3]);
title('Best value found so far'); legend('Location', 'southwest');
exportgraphics(fig, fullfile(figdir, 'p1_4_subgradient.png'), 'Resolution', 200);
