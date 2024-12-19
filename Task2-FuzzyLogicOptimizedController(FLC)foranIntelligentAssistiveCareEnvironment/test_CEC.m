% Test Benchmark Functions (Manually Created)

% Input for D = 2 dimensions
x = [1, 1];  

% Test Elliptic Function
f_elliptic = ellipticFunction(x);
fprintf('Elliptic Function value at [%s] = %.4f\n', num2str(x), f_elliptic);

% Test Griewank Function
f_griewank = griewankFunction(x);
fprintf('Griewank Function value at [%s] = %.4f\n', num2str(x), f_griewank);
x = ones(1, 10); % 10-dimensional input
f_elliptic = ellipticFunction(x);
fprintf('Elliptic Function value at [%s] = %.4f\n', num2str(x), f_elliptic);

f_griewank = griewankFunction(x);
fprintf('Griewank Function value at [%s] = %.4f\n', num2str(x), f_griewank);
