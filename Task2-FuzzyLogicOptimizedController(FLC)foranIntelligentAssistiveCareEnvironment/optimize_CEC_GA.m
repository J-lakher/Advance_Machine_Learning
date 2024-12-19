% GA Optimization on Elliptic and Griewank Functions with Convergence Graphs

% Define problem dimensions
D = 2;  % You can increase to 10 later

% Bounds for the search space
lowerBound = -5 * ones(1, D); % Lower bound
upperBound = 5 * ones(1, D);  % Upper bound

% GA options
optionsGA = optimoptions('ga', 'PopulationSize', 50, 'MaxGenerations', 100, ...
    'OutputFcn', @gaOutputFcn, 'Display', 'iter');

% Global variable to store convergence data
global gaBestFitness;
gaBestFitness = [];

% Optimize Elliptic Function
fprintf('Optimizing Elliptic Function using GA...\n');
[bestSolution_elliptic, bestFitness_elliptic] = ga(@ellipticFunction, D, [], [], [], [], ...
    lowerBound, upperBound, [], optionsGA);

% Plot Convergence Graph for Elliptic Function
figure;
plot(gaBestFitness, 'LineWidth', 2);
xlabel('Generation');
ylabel('Best Fitness Value');
title('GA Convergence - Elliptic Function');
grid on;

% Reset convergence data for Griewank Function
gaBestFitness = [];

% Optimize Griewank Function
fprintf('Optimizing Griewank Function using GA...\n');
[bestSolution_griewank, bestFitness_griewank] = ga(@griewankFunction, D, [], [], [], [], ...
    lowerBound, upperBound, [], optionsGA);

% Plot Convergence Graph for Griewank Function
figure;
plot(gaBestFitness, 'LineWidth', 2);
xlabel('Generation');
ylabel('Best Fitness Value');
title('GA Convergence - Griewank Function');
grid on;

% Display results
fprintf('GA Best Fitness for Elliptic Function: %.4f\n', bestFitness_elliptic);
fprintf('GA Best Fitness for Griewank Function: %.4f\n', bestFitness_griewank);

% Perform 15 runs for statistical analysis
numRuns = 15;
bestResults_elliptic = zeros(numRuns, 1);
bestResults_griewank = zeros(numRuns, 1);

for i = 1:numRuns
    % Elliptic Function
    [~, bestFitness] = ga(@ellipticFunction, D, [], [], [], [], lowerBound, upperBound, [], optionsGA);
    bestResults_elliptic(i) = bestFitness;

    % Griewank Function
    [~, bestFitness] = ga(@griewankFunction, D, [], [], [], [], lowerBound, upperBound, [], optionsGA);
    bestResults_griewank(i) = bestFitness;
end

% Calculate and display statistics
meanFitness_elliptic = mean(bestResults_elliptic);
stdFitness_elliptic = std(bestResults_elliptic);
fprintf('Elliptic Function - Mean Fitness: %.4f, Standard Deviation: %.4f\n', ...
    meanFitness_elliptic, stdFitness_elliptic);

meanFitness_griewank = mean(bestResults_griewank);
stdFitness_griewank = std(bestResults_griewank);
fprintf('Griewank Function - Mean Fitness: %.4f, Standard Deviation: %.4f\n', ...
    meanFitness_griewank, stdFitness_griewank);

% Corrected Output Function to Track Best Fitness Across Generations
function [state, options, optchanged] = gaOutputFcn(options, state, flag)
    global gaBestFitness;
    optchanged = false; % No options changed
    
    if strcmp(flag, 'iter') % During each generation
        gaBestFitness(end+1) = state.Best(end); % Store best fitness value
    end
end

% Elliptic Function Definition
function y = ellipticFunction(x)
    D = numel(x);
    y = sum((1e6).^(linspace(0, 1, D)) .* (x.^2));
end

% Griewank Function Definition
function y = griewankFunction(x)
    part1 = sum(x.^2) / 4000;
    part2 = prod(cos(x ./ sqrt(1:numel(x))));
    y = part1 - part2 + 1;
end
