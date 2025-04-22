% Main program setup
% Rev 2.0, April 2025
% Developed By Prof. Adel Ben Abdennourclear variables;
% ayhem123@yahoo.com
% Citation: BenAbdennour, A. 
% An Enhanced Team-Oriented Swarm Optimization Algorithm (ETOSO) for
% Robust and Efficient High-Dimensional Search. 
% Biomimetics 2025,10, 222. https://doi.org/10.3390/biomimetics10040222

%
% Objective:
% This script evaluates and compares the performance of multiple optimization
% algorithms on a set of benchmark functions. It tracks metrics such as best
% fitness, average performance, speed, and consistency across multiple
% replications. Results are saved to an Excel file for further analysis.
%
% Key Features:
% - Supports multiple optimization algorithms and benchmark functions.
% - Tracks performance, speed, and consistency metrics.
% - Includes an early stopping mechanism based on relative improvement.
% - Generates plots and saves results to an Excel file.
%
% Input Parameters:
% - plotfcn: Toggle plotting (1: disable, 2: enable).
% - nr: Number of replications.
% - D: Dimension of the search space.
% - tmpfe: Maximum function evaluations (FE).
% - useRelStop: Enable/disable early stopping based on relative improvement.
% - relTol: Tolerance threshold for relative improvement.
% - evalWindow: Evaluation window for stopping check.
%
% Output:
% - Excel file containing results for each algorithm and function.
% - Rankings based on performance, speed, and consistency.
%
% Example Usage:
% Run the script with default parameters to evaluate algorithms on benchmark
% functions and generate results.
%
% Notes:
% - Ensure all referenced algorithms and benchmark functions are available.
% - Results are saved to an Excel file with a timestamped filename.
%



close all;
clc;
warning off;
format compact;
format longE;

% Set main program parameters
plotfcn = 1;              % Set to 2 to enable plotting, 1 to disable
nr = 2;                   % Number of replications
D = 2;                   % Dimension
tmpfe = 50 * D;         % Max function evaluations (FE)
useRelStop = false;        % Toggle relative improvement stopping criterion
relTol = 1e-6;
evalWindow = tmpfe/2;        % Evaluation window for stopping check

functnames = {'f1','f2','f3','f4','f5','f6','f7','f8','f9','f10','f11','f12','f13','f14','f15'};
algorithms = {'ETOSO','BAT','BEE','BOA','CSA','CS','DE','EHO','FA','FDA','FPA','GOA','GSA','GWO','HHO','MFO','MKA',...
    'PDO','RRO','ROA','SCA','SOA','SMA','SSA','TLBO','WOA'}; 

%algorithms = {'TOSO','ETOSO'}%, 'ETOSO2', 'BAT2', 'DE2', 'BEE2', 'BOA2', 'CS2','EHO2','FA2'};

fc = length(functnames);
num_algorithms = length(algorithms);
all_results = cell(num_algorithms, 1);
stopping_points = zeros(num_algorithms, fc);  % Store actual stopping points for each algorithm

fprintf('****  Dimension : %10d\n', D);
known_minima = [0, 0, 0, 0, -418.9829 * D, 0, 0, 0, 0, 0, -140, 390, -330, -180, 0];
fes = repmat(tmpfe, 1, fc);

% Performance tracking variables
best_fitness = zeros(num_algorithms, fc);
avg_performance = zeros(num_algorithms, fc);
std_performance = zeros(num_algorithms, fc);
avg_speed = zeros(num_algorithms, fc);
std_speed = zeros(num_algorithms, fc);

% Loop through each function
for n_func = 1:fc
    fprintf('Optimizing Function No : %10d\n', n_func);
    base_seed = 100 * n_func;
    range = determineRange(n_func);  % Get range for the current function

    % Loop over algorithms
    for n_alg = 1:num_algorithms
        all_results{n_alg} = zeros(1, fes(n_func));  % Preallocate to store sum of fitness over replications
        fitness = zeros(1, nr);  % To store best fitness per replication
        etime = zeros(1, nr);    % To store execution time per replication

        for r = 1:nr
            rng(base_seed + r);  % Set seed for each replication
            tic;  % Start timing
            
            % Run the selected algorithm with the relative stopping parameters
            algorithm_func = str2func(algorithms{n_alg});
         
% Run the selected algorithm with the relative stopping parameters
[yout, te, tr, total_evals] = algorithm_func(n_func, D, range, fes(n_func), useRelStop, relTol, evalWindow);

            % Ensure `tr` is the same length as `fes(n_func)`
            if length(tr) < fes(n_func)
                tr = [tr, repmat(tr(end), 1, fes(n_func) - length(tr))];  % Pad with last value if too short
            elseif length(tr) > fes(n_func)
                tr = tr(1:fes(n_func));  % Truncate if too long
            end

            % Accumulate fitness values over replications
            all_results{n_alg} = all_results{n_alg} + tr;

            % Store replication results
% Store replication results
if isvector(yout)  % Check if `yout` is a vector
    fitness(r) = yout(end);  % Use the last value if it's an array
else
    fitness(r) = yout;  % Directly assign if `yout` is already scalar
end
etime(r) = toc;     % Elapsed time up to stopping point
   % Track stopping point for this algorithm
    stopping_points(n_alg, n_func) = min(fes(n_func), total_evals);
            % Track stopping point for this algorithm if it stopped early
% Ensure `total_evals` is a scalar by taking only the last element if it's an array
if isvector(total_evals)
    stopping_points(n_alg, n_func) = max(stopping_points(n_alg, n_func), total_evals(end));
else
    stopping_points(n_alg, n_func) = max(stopping_points(n_alg, n_func), total_evals);
end
        end

        % Compute the average fitness across replications for each evaluation point
        all_results{n_alg} = all_results{n_alg} / nr;
        best_fitness(n_alg, n_func) = min(fitness);
        avg_performance(n_alg, n_func) = mean(fitness);
        std_performance(n_alg, n_func) = std(fitness);
        avg_speed(n_alg, n_func) = mean(etime);
        std_speed(n_alg, n_func) = std(etime);
    end

    % Plotting the average fitness
 % Plotting the average fitness
if plotfcn == 2
    markers = {'o', '+', '*', '.', 'x', 's', 'd', '^', 'v', '>', '<', 'p', 'h'};
    line_styles = {'-', '--', ':', '-.', '-', '--', ':', '-.'};  % Define line styles
    algorithms_per_subplot = 7;
    num_subplots = ceil(num_algorithms / algorithms_per_subplot);

    figure('Position', [100, 100, 1200, 800]);
    for subplot_idx = 1:num_subplots
        subplot(num_subplots, 1, subplot_idx);
        hold on;

        title(sprintf('Function: %s - Algorithms Group %d', functnames{n_func}, subplot_idx));

        % Determine algorithm range for this subplot
        start_idx = (subplot_idx - 1) * algorithms_per_subplot + 1;
        end_idx = min(start_idx + algorithms_per_subplot - 1, num_algorithms);

        % Plot results for each algorithm in the current group
        for n_alg = start_idx:end_idx
            tr = all_results{n_alg};  % Average fitness over replications
            te = 1:length(tr);        % X-axis for evaluations
            marker_idx = mod(n_alg - 1, length(markers)) + 1;
            line_style_idx = mod(n_alg - 1, length(line_styles)) + 1;  % Cycle through line styles

            % Plot the average fitness over all replications
            plot(te, tr, 'DisplayName', algorithms{n_alg}, ...
                'Marker', markers{marker_idx}, ...
                'LineStyle', line_styles{line_style_idx}, ...  % Apply different line styles
                'MarkerIndices', round(linspace(1, length(te), 50)), ...
                'LineWidth', 2.5);

            % Identify stopping point and add debug output
            stop_eval = stopping_points(n_alg, n_func);  % Stopping point for this algorithm
            % fprintf('Algorithm: %s, Function: %s, Stopping Point: %d, Fitness at Stop: %.5f\n', ...
                    % algorithms{n_alg}, functnames{n_func}, stop_eval, tr(stop_eval));

            % Ensure the stopping point is within bounds
            if stop_eval < fes(n_func) && stop_eval <= length(tr)
                plot(stop_eval, tr(stop_eval), 'ro', 'MarkerSize', 8, ...
                    'MarkerFaceColor', 'r', 'HandleVisibility', 'off');  % Red circle marker
            end
        end

        xlabel('Function Evaluations');
        ylabel('Average Best Fitness');
        legend('show');
        grid on;
    end
end

end

% Calculate ranks based on performance, speed, and consistency
[performance_rank, speed_rank, consistency_rank] = rank_algorithms(avg_performance, std_performance, avg_speed, known_minima);

% Additional functions for ranking and output (these are required for the final output and rankings)

%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%

% Calculate average ranks across functions
avg_performance_rank = mean(performance_rank, 2); % Average performance rank
avg_speed_rank = mean(speed_rank, 2); % Average speed rank
avg_consistency_rank = mean(consistency_rank, 2); % Average consistency rank

% Custom function for integer ranking without averaging ties





%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Prepare Excel file name with current timestamp
current_time = datetime('now');
filename = sprintf('results_%02d_%02d%02d.xlsx', current_time.Day, current_time.Hour, current_time.Minute);

% Initialize the cell array for the Excel data
%output = cell((length(algorithms) + 1) * 7 + 1, length(functnames) + 2); % 7 metrics + 1 for headers
output = cell((length(algorithms) + 1) * 2 + 1, length(functnames) + 2); % 7 metrics + 1 for headers

% Set headers
output(1, :) = [{'Algorithm', 'Metric'}, functnames];

% Insert known minimums
output(2, :) = [{''}, {'Known Minimum'}, num2cell(known_minima)];

% Loop through the algorithms and insert the metrics
for i = 1:length(algorithms)
    algo = algorithms{i};

   % start_row = (i - 1) * 7 + 3; % Row index based on number of metrics
    start_row = (i - 1) * 2 + 3; % Row index based on number of metrics

    % Set algorithm name and metrics
    output{start_row, 1} = algo; % Algorithm name

    output{start_row, 2} = 'Best';
    output(start_row, 3:end) = num2cell(best_fitness(i, :));

    output{start_row + 1, 2} = 'Avg Performance';
    output(start_row + 1, 3:end) = num2cell(avg_performance(i, :));

    % output{start_row + 2, 2} = 'Std Performance';
    % output(start_row + 2, 3:end) = num2cell(std_performance(i, :));
    % 
    % output{start_row + 3, 2} = 'Performance Rank';
    % output(start_row + 3, 3:end) = num2cell(performance_rank(i, :));
    % 
    % output{start_row + 4, 2} = 'Avg Speed';
    % output(start_row + 4, 3:end) = num2cell(avg_speed(i, :));
    % 
    % output{start_row + 5, 2} = 'Std Speed';
    % output(start_row + 5, 3:end) = num2cell(std_speed(i, :));
    % 
    % output{start_row + 6, 2} = 'Speed Rank'; % Ensure speed rank is included
    % output(start_row + 6, 3:end) = num2cell(speed_rank(i, :));
end

% Write this output to an Excel file (optional)
% xlswrite(filename, output);

% Rank for each metric using custom ranking function
performance_rank_final = custom_integer_ranking(avg_performance_rank);
speed_rank_final = custom_integer_ranking(avg_speed_rank);
consistency_rank_final = custom_integer_ranking(avg_consistency_rank);

% Number of algorithms
n_alg = size(performance_rank, 1);

% Prepare the output for rankings
rankings_output = cell(n_alg + 1, 4); % Columns: Algorithm, Performance Rank, Speed Rank, Consistency Rank
rankings_output(1, :) = {'Algorithm', 'Performance Rank', 'Speed Rank', 'Consistency Rank'}; % Header

% Populate the rankings
for i = 1:n_alg
    rankings_output{i + 1, 1} = algorithms{i}; % Use the actual algorithm name
    rankings_output{i + 1, 2} = performance_rank_final(i); % Performance rank
    rankings_output{i + 1, 3} = speed_rank_final(i); % Speed rank
    rankings_output{i + 1, 4} = consistency_rank_final(i); % Consistency rank
end
% Write to Excel
%   writecell(output, filename);

% Write main data to Excel
writecell(output, filename);
% Now create the additional information to append to the end
additional_info = {
    '', 'Additional Information', sprintf('Replications: %d, Dimension: %d, FE: %d', nr, D, fes(1))
    };
% Write the additional information to the next available row
startRow = size(output, 1) + 1; % Start after the last written row
writecell(additional_info, filename, 'Sheet', 1, 'Range', sprintf('A%d', startRow));


% Write the rankings_output to the Excel file (add this to your existing writing logic)
startRow = size(output, 1) + 5; % Adjust appropriately to write after previous data
writecell(rankings_output, filename, 'Sheet', 1, 'Range', sprintf('A%d', startRow));

% Notify that rankings have been added to the Excel file
fprintf('Performance, Speed, and Consistency rankings written to %s\n', filename);







%%%%%%%%%%%%
function rank_final = custom_integer_ranking(values)
n = length(values);
[sorted_values, sorted_indices] = sort(values); % Sort values and keep track of original indices
rank_final = zeros(size(values)); % Initialize rank output
current_rank = 1; % Start with rank 1

i = 1; % Iterator for sorted values
while i <= n
    % Check how many times the current value appears (for tied ranks)
    j = i; % Start counting ties
    while j < n && sorted_values(j + 1) == sorted_values(i)
        j = j + 1; % Move to the next element
    end

    % Assign the current rank to all tied elements
    rank_final(sorted_indices(i:j)) = current_rank;

    % Increment the current rank by the number of tied values
    current_rank = current_rank + (j - i + 1);
    i = j + 1; % Move past the tied elements
end
end



%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%
% Supporting function to determine the range for each function

% Rank algorithms based on performance, speed, and consistency
function [performance_rank, speed_rank, consistency_rank] = rank_algorithms(avg_performance, std_performance, avg_speed, known_minima)
    n_alg = size(avg_performance, 1);
    n_func = size(avg_performance, 2);
    performance_rank = zeros(n_alg, n_func);
    speed_rank = zeros(n_alg, n_func);
    consistency_rank = zeros(n_alg, n_func);

    for i = 1:n_func
        diff_from_min = abs(avg_performance(:, i) - known_minima(i));
        performance_rank(:, i) = assign_tied_rank(diff_from_min);

        speed_rank(:, i) = assign_tied_rank(avg_speed(:, i));
        consistency_rank(:, i) = assign_tied_rank(std_performance(:, i));
    end
end

%%%%%%%%%%%%%%
% Helper function to assign ranks with ties
function ranks = assign_tied_rank(values)
    [~, sorted_indices] = sort(values);
    ranks = zeros(size(values));
    current_rank = 1;
    num_values = numel(values);

    for j = 1:num_values
        if j == 1 || values(sorted_indices(j)) ~= values(sorted_indices(j-1))
            ranks(sorted_indices(j)) = current_rank;
            current_rank = current_rank + 1;
        else
            ranks(sorted_indices(j)) = current_rank - 1;
        end
    end
end


function range = determineRange(funcIndex)
    switch funcIndex
        case 1, range = [-100, 100];
        case 2, range = [-5, 10];
        case 3, range = [-10, 10];
        case 4, range = [-10, 10];
        case 5, range = [-500, 500];
        case 6, range = [-30, 30];
        case 7, range = [-32, 32];
        case 8, range = [-5.12, 5.12];
        case 9, range = [-600, 600];
        case 10, range = [-100, 100];
        case 11, range = [-32, 32];
        case 12, range = [-100, 100];
        case 13, range = [-5, 5];
        case 14, range = [-600, 600];
        case 15, range = [-600, 600];
    end
end
