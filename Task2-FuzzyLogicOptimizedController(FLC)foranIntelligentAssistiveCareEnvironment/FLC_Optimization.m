% Dataset: Temperature (°C) -> Desired HeaterPower (%)
fis = readfis('Temperature_HeaterControl_FLC.fis');
data = [
    15, 90;  % At 15°C, desired HeaterPower is 90%
    18, 70;  % At 18°C, desired HeaterPower is 70%
    22.5, 50; % At 22.5°C, desired HeaterPower is 50%
    26, 30;  % At 26°C, desired HeaterPower is 30%
    30, 10;  % At 30°C, desired HeaterPower is 10%
];

% Plot the dataset
figure;
plot(data(:,1), data(:,2), 'bo-', 'LineWidth', 2);
xlabel('Temperature (°C)');
ylabel('Desired HeaterPower (%)');
title('Dataset for FLC Optimization');
grid on;

% Test the FIS with a sample input
input = 22.5; % Example input temperature
output = evalfis(fis, input); % Evaluate FIS
disp(['Output HeaterPower for Temperature = ', num2str(input), '°C: ', num2str(output), '%']);

%% GA Parameters
populationSize = 20;   % Number of individuals in the population
numGenerations = 50;   % Number of generations
mutationRate = 0.1;    % Probability of mutation
numParams = 18;        % Number of parameters to optimize (6 MFs * 3 parameters each)

% Define bounds for chromosome values (membership function parameters)
lowerBounds = [15, 15, 18, 18, 22, 26, 24, 30, 30, 0, 0, 25, 20, 50, 70, 60, 80, 100];
upperBounds = [15, 21, 25, 20, 26, 30, 25, 30, 30, 0, 50, 50, 40, 70, 100, 80, 100, 100];

% Initialize random population within bounds
population = rand(populationSize, numParams) .* (upperBounds - lowerBounds) + lowerBounds;

% Track the best fitness history
bestFitnessHistory = zeros(numGenerations, 1);

%% Main GA Loop
for generation = 1:numGenerations
    % Step 1: Evaluate fitness for each chromosome
    fitnessValues = zeros(populationSize, 1);
    for i = 1:populationSize
        fitnessValues(i) = evaluateFitness(population(i, :), data, fis);
    end
    
    % Step 2: Selection - Sort population based on fitness (elitism)
    [~, sortedIndices] = sort(fitnessValues, 'descend');
    population = population(sortedIndices, :);
    bestFitnessHistory(generation) = fitnessValues(sortedIndices(1));
    
    % Step 3: Crossover - Combine top individuals to produce offspring
    newPopulation = population(1:2, :); % Elitism: keep the top 2 individuals
    while size(newPopulation, 1) < populationSize
        % Select two parents randomly from the top 50%
        parent1 = population(randi([1, populationSize/2]), :);
        parent2 = population(randi([1, populationSize/2]), :);
        
        % Single-point crossover
        point = randi([1, numParams-1]);
        child1 = [parent1(1:point), parent2(point+1:end)];
        child2 = [parent2(1:point), parent1(point+1:end)];
        
        % Add offspring to new population
        newPopulation = [newPopulation; child1; child2];
    end
    
    % Step 4: Mutation - Apply random changes to offspring
    for i = 3:populationSize % Skip the top 2 (elitism)
        if rand < mutationRate
            mutationPoint = randi([1, numParams]);
            newPopulation(i, mutationPoint) = lowerBounds(mutationPoint) + ...
                rand * (upperBounds(mutationPoint) - lowerBounds(mutationPoint));
        end
    end
    
    % Replace the old population with the new one
    population = newPopulation;
    
    % Display progress
    disp(['Generation ', num2str(generation), ': Best Fitness = ', num2str(bestFitnessHistory(generation))]);
end

% Plot the Best Fitness Over Generations
figure;
plot(1:numGenerations, bestFitnessHistory, 'LineWidth', 2);
xlabel('Generation');
ylabel('Best Fitness');
title('GA Optimization Progress');
grid on;

% Apply the Best Chromosome to the FIS
bestChromosome = population(1, :);
optimizedFis = applyChromosomeToFIS(bestChromosome, fis);

% Test the Optimized FIS
testInput = 22.5; % Example input
optimizedOutput = evalfis(optimizedFis, testInput);
disp(['Optimized Output for Temperature = ', num2str(testInput), '°C: ', num2str(optimizedOutput), '%']);

%% Function Definitions
function fitness = evaluateFitness(chromosome, data, fis)
    % Apply the chromosome to the FLC
    optimizedFis = applyChromosomeToFIS(chromosome, fis);
    
    % Calculate the Mean Squared Error (MSE)
    mse = 0;
    for i = 1:size(data, 1)
        x = data(i, 1); % Input temperature
        y_true = data(i, 2); % Desired heater power
        y_pred = evalfis(optimizedFis, x); % FLC output
        mse = mse + (y_true - y_pred)^2;
    end
    mse = mse / size(data, 1);
    
    % Fitness is the inverse of MSE
    fitness = 1 / mse;
end

function optimizedFis = applyChromosomeToFIS(chromosome, fis)
    % Update the membership functions of the FIS based on the chromosome
    % Assumes FIS has 2 variables (Temperature, HeaterPower), each with 3 triangular MFs
    % Chromosome contains 18 parameters (6 MFs * 3 parameters each)

    % Reshape chromosome into matrix form
    mfParams = reshape(chromosome, [3, 6])';

    % Sort each set of MF parameters to satisfy a <= b <= c
    for i = 1:6
        mfParams(i, :) = sort(mfParams(i, :));
    end

    % Update Temperature MFs
    for i = 1:3
        fis.Inputs(1).MembershipFunctions(i).Parameters = mfParams(i, :);
    end

    % Update HeaterPower MFs
    for i = 1:3
        fis.Outputs(1).MembershipFunctions(i).Parameters = mfParams(i+3, :);
    end

    optimizedFis = fis;
end
