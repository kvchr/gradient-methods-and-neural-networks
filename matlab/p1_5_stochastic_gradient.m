% Part 1, Problem 5: stochastic gradient descent
% f(x) = 1/4 * sum_{i=1..4} [(x1 - a_i)^2 + (x2 - b_i)^2]
% Its gradient is the sum of four terms d_i = 1/2 * [x1 - a_i; x2 - b_i].
% Gradient descent uses all four; stochastic gradient descent uses one random d_i per iteration.

clc; clearvars; close all;
figdir = fullfile(fileparts(mfilename('fullpath')), '..', 'figures');
rng(0);

a = randi([0, 10], 1, 4);              % the four points (a_i, b_i)
b = randi([0, 10], 1, 4);
C = [a; b];
f = @(x) 1/4 * sum((x(1) - a).^2 + (x(2) - b).^2);
d = @(x, i) 1/2 * (x - C(:, i));       % one term of the gradient
x_star = mean(C, 2);                   % f = |x - x*|^2 + const, so the minimiser is the mean point
f_star = f(x_star);
s = sum(var(C, 1, 2));                 % mean squared distance of the points from x*
fprintf('points (a_i, b_i): %s\n', mat2str(C'));
fprintf('minimum: x* = (%.2f, %.2f), f* = %.4f, mean squared distance of the points from x*: s = %.4f\n', x_star, f_star, s);

budget = 400;                          % number of d_i evaluations allowed
runs = 1000;                           % random starting points in [0, 10]^2
methods = {'GD, \alpha = 0.1', 'SGD, \alpha = 0.1', 'SGD, \alpha = 0.02', 'SGD, \alpha_k = 2/(k+1)'};
err = zeros(numel(methods), budget + 1, runs);     % f - f* after each d_i evaluation
x0s = 10 * rand(2, runs);
for r = 1:runs
    x0 = x0s(:, r);
    % Gradient descent: 4 evaluations per iteration
    x = x0; e = zeros(1, budget + 1); e(1) = f(x) - f_star;
    for k = 1:budget/4
        g = d(x, 1) + d(x, 2) + d(x, 3) + d(x, 4);
        x = x - 0.1 * g;
        e(4*k-2:4*k) = e(4*k-3);       % the point moves only after the 4th evaluation
        e(4*k+1) = f(x) - f_star;
    end
    err(1, :, r) = e;
    % Stochastic gradient descent: 1 evaluation per iteration, i chosen uniformly at random
    for m = 2:4
        x = x0; e = zeros(1, budget + 1); e(1) = f(x) - f_star;
        for k = 1:budget
            switch m
                case 2, alpha = 0.1;
                case 3, alpha = 0.02;
                case 4, alpha = 2 / (k + 1);
            end
            x = x - alpha * d(x, randi(4));
            e(k + 1) = f(x) - f_star;
        end
        err(m, :, r) = e;
    end
end

avg = max(mean(err, 3), 0);            % average over the starting points
fprintf('\nf - f* averaged over %d starting points:\n', runs);
fprintf('%-24s %10s %10s %10s %10s   %s\n', 'method', '40 evals', '100 evals', '200 evals', '400 evals', 'evals until below 0.05');
for m = 1:numel(methods)
    reach = find(avg(m, :) < 0.05, 1) - 1;
    if isempty(reach), reach_txt = 'never'; else, reach_txt = sprintf('%d', reach); end
    fprintf('%-24s %10.2e %10.2e %10.2e %10.2e   %s\n', strrep(methods{m}, '\', ''), ...
        avg(m, 41), avg(m, 101), avg(m, 201), avg(m, 401), reach_txt);
end

% Theory checks
% Fixed step: e_{k+1} = (1 - alpha/2) e_k + (alpha/2)(c_i - x*), so in the long run E|e|^2 = alpha*s/(4 - alpha).
for alpha = [0.1 0.02]
    x = x_star; acc = 0; n_long = 200000;
    for k = 1:n_long
        x = x - alpha * d(x, randi(4));
        acc = acc + sum((x - x_star).^2);
    end
    fprintf('SGD, alpha = %.2f: long-run mean of f - f*: predicted %.4f, measured over %d iterations %.4f\n', ...
        alpha, alpha * s / (4 - alpha), n_long, acc / n_long);
end
% alpha_k = 2/(k+1): x_k is the average of x0 and the k sampled points, so E|e_k|^2 = (|x0 - x*|^2 + k s) / (k+1)^2.
k = budget;
pred = mean((sum((x0s - x_star).^2, 1) + k * s) / (k + 1)^2);
fprintf('SGD, alpha_k = 2/(k+1): f - f* after %d iterations: predicted %.4f, measured %.4f\n', k, pred, avg(4, k + 1));

%% Figures
fig = figure('Color', 'w', 'Position', [100 100 1150 440]);
subplot(1, 2, 1);
cols = lines(numel(methods));
for m = 1:numel(methods)
    semilogy(0:budget, max(avg(m, :), 1e-16), 'LineWidth', 1.4, 'Color', cols(m, :), 'DisplayName', methods{m}); hold on;
end
grid on; ylim([1e-4 100]);
xlabel('number of d_i evaluations'); ylabel(sprintf('f(x) - f^*  (mean of %d starts)', runs));
title('Cost counted in evaluations of one term d_i'); legend('Location', 'southwest');
subplot(1, 2, 2);
rng(1);
x0 = [9; 9];
[X, Y] = meshgrid(-1:0.1:11, -1:0.1:11);
Z = zeros(size(X));
for i = 1:4, Z = Z + 1/4 * ((X - a(i)).^2 + (Y - b(i)).^2); end
contour(X, Y, Z, 20, 'LineColor', [0.8 0.8 0.8], 'HandleVisibility', 'off'); hold on;
for m = [2 4]
    x = x0; pth = x0';
    for k = 1:budget
        if m == 2, alpha = 0.1; else, alpha = 2 / (k + 1); end
        x = x - alpha * d(x, randi(4));
        pth(end+1, :) = x'; %#ok<AGROW>
    end
    plot(pth(:, 1), pth(:, 2), '.-', 'Color', cols(m, :), 'MarkerSize', 5, 'DisplayName', methods{m});
end
plot(a, b, 'ks', 'MarkerFaceColor', [0.6 0.6 0.6], 'DisplayName', 'points (a_i, b_i)');
plot(x_star(1), x_star(2), 'kp', 'MarkerSize', 13, 'MarkerFaceColor', 'k', 'DisplayName', 'minimum');
axis equal; xlim([-1 11]); ylim([-1 11]);
xlabel('x_1'); ylabel('x_2'); title('One SGD run from (9, 9), 400 iterations'); legend('Location', 'northwest');
set(findall(fig, 'Type', 'axes'), 'Toolbar', []);   % keep the axes toolbar out of the image
exportgraphics(fig, fullfile(figdir, 'p1_5_sgd.png'), 'Resolution', 200);
