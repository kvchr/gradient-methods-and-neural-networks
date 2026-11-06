% Part 2, Problem 1: a network with one hidden ReLU layer and a linear output
% that represents the piecewise-linear function
%   g(x) = -x - 2  (x < -2),   x + 2  (-2 <= x < -1),   -x  (-1 <= x < 0),   x  (x >= 0)
% (A) exact construction with 4 hidden neurons, (B) a 120-neuron network trained by gradient descent.

clc; clearvars; close all;
figdir = fullfile(fileparts(mfilename('fullpath')), '..', 'figures');

g = @(x) (x < -2) .* (-x - 2) + (-2 <= x & x < -1) .* (x + 2) + (-1 <= x & x < 0) .* (-x) + (x >= 0) .* x;
x_test = linspace(-3, 3, 1001);

%% (A) Exact construction: g(x) = ReLU(x) + ReLU(-x) - 2 ReLU(-x - 1) + 2 ReLU(-x - 2)
W1 = [1; -1; -1; -1];        % input -> hidden weights
b1 = [0; 0; -1; -2];         % hidden biases (the kinks are at x = 0, 0, -1, -2)
W2 = [1 1 -2 2];             % hidden -> output weights
b2 = 0;                      % output bias
net4 = feedforwardnet(4);
net4.inputs{1}.processFcns = {};          % no input/output scaling, so the weights are used as written
net4.outputs{2}.processFcns = {};
net4.layers{1}.transferFcn = 'poslin';    % ReLU
net4 = configure(net4, x_test, g(x_test));
net4.IW{1, 1} = W1; net4.b{1} = b1;
net4.LW{2, 1} = W2; net4.b{2} = b2;
y4 = net4(x_test);
fprintf('(A) 4 ReLU neurons, weights set by hand: max |error| on [-3, 3] = %.2e\n', max(abs(y4 - g(x_test))));

%% (B) 120 ReLU neurons trained with plain gradient descent (traingd)
rng(42);                     % seed for the weight initialisation and the train/validation/test split
x_train = linspace(-3, 3, 3000);
net = fitnet(120, 'traingd');
net.layers{1}.transferFcn = 'poslin';
net.trainParam.epochs = 1200;
net.trainParam.showWindow = false;
[net, tr] = train(net, x_train, g(x_train));
y_pred = net(x_test);
err = abs(y_pred - g(x_test));
fprintf('(B) 120 ReLU neurons, traingd: stopped after %d epochs (%s)\n', tr.num_epochs, tr.stop);
fprintf('    max |error| = %.4f, mean |error| = %.4f on 1001 test points in [-3, 3]\n', max(err), mean(err));
kinks = [-2 -1 0];
fprintf('    error at the kinks x = -2, -1, 0: %s; mean |error| elsewhere (|x - kink| > 0.1): %.4f\n', ...
    mat2str(net(kinks) - g(kinks), 3), mean(err(min(abs(x_test' - kinks), [], 2)' > 0.1)));

%% Figure
fig = figure('Color', 'w', 'Position', [100 100 1150 420]);
subplot(1, 2, 1);
plot(x_test, g(x_test), 'k-', 'LineWidth', 2.2, 'DisplayName', 'g(x)'); hold on;
plot(x_test, y4, 'c--', 'LineWidth', 1.6, 'DisplayName', '4 neurons, exact');
plot(x_test, y_pred, 'r:', 'LineWidth', 1.8, 'DisplayName', '120 neurons, trained');
grid on; xlabel('x'); ylabel('output'); legend('Location', 'north'); title('Target and networks');
subplot(1, 2, 2);
plot(x_test, y_pred - g(x_test), 'r-', 'LineWidth', 1.2); grid on;
xlabel('x'); ylabel('error'); title('Error of the trained 120-neuron network');
exportgraphics(fig, fullfile(figdir, 'p2_1_relu_network.png'), 'Resolution', 200);
