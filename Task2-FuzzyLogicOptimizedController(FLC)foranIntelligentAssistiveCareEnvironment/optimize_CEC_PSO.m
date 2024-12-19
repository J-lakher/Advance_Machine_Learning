% PSO Optimization on Elliptic and Griewank Functions with Convergence Graphs

% Define problem dimensions
D = 2;  % You can increase to 10 later

% Bounds for the search space
lowerBound = -5 * ones(1, D); % Lower bound
upperBound = 5 * ones(1, D);  % Upper bound

% PSO options
global psoBestFitness; % Global variable to store convergence data
psoBestFitness = [];   % Reset convergence data

optionsPSO = optimoptions('particleswarm', ...
    'SwarmSize', 50, 'MaxIterations', 100, ...
    'OutputFcn', @psoOutputFcn, 'Display', 'iter');

%% Optimize Elliptic Function
fprintf('Optimizing Elliptic Function using PSO...\n');
psoBestFitness = []; % Reset convergence data for Elliptic function
[bestSolution_elliptic, bestFitness_elliptic] = particleswarm(@ellipticFunction, D, ...
    lowerBound, upperBound, optionsPSO);

% Plot Convergence Graph for Elliptic Function
figure;
plot(psoBestFitness, 'LineWidth', 2);
xlabel('Iteration');
ylabel('Best Fitness Value');
title('PSO Convergence - Elliptic Function');
grid on;

%% Optimize Griewank Function
fprintf('Optimizing Griewank Function using PSO...\n');
psoBestFitness = []; % Reset convergence data for Griewank function
[bestSolution_griewank, bestFitness_griewank] = particleswarm(@griewankFunction, D, ...
    lowerBound, upperBound, optionsPSO);

% Plot Convergence Graph for Griewank Function
figure;
plot(psoBestFitness, 'LineWidth', 2);
xlabel('Iteration');
ylabel('Best Fitness Value');
title('PSO Convergence - Griewank Function');
grid on;

%% Display Results
fprintf('PSO Best Fitness for Elliptic Function: %.4f\n', bestFitness_elliptic);
fprintf('PSO Best Fitness for Griewank Function: %.4f\n', bestFitness_griewank);

%% PSO Output Function to Track Convergence
function stop = psoOutputFcn(optimValues, state)
    % Global variable to track the best fitness
    global psoBestFitness;
    
    % Add the current best fitness value
    psoBestFitness(end+1) = optimValues.bestfval; % Correct field name is "bestfval"
    
    % Do not stop the algorithm
    stop = false;
end


%% Elliptic Function Definition
function y = ellipticFunction(x)
    D = numel(x);
    y = sum((1e6).^(linspace(0, 1, D)) .* (x.^2));
end

%% Griewank Function Definition
function y = griewankFunction(x)
    part1 = sum(x.^2) / 4000;
    part2 = prod(cos(x ./ sqrt(1:numel(x))));
    y = part1 - part2 + 1;
end
