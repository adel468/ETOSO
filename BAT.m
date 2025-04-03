function [res, varargout] = BAT(I, D, range, FE, useRelStop, relTol, evalWindow)
    % Parameters based on the paper's multi-strategy approach
    pop_size = 30;             % Population size
    A = 0.95;                  % Initial loudness
    r0 = 0.9;                  % Initial pulse rate
    Qmin = 0;                  % Minimum frequency
    Qmax = 5;                  % Maximum frequency (smaller range for mixBA)
    alpha = 0.99;              % Loudness decay rate
    gamma = 0.9;               % Pulse rate increase rate
    strategies = 8;            % Number of movement strategies

    % Initialize population
    pos = rand(pop_size, D) .* (range(2) - range(1)) + range(1);
    velocity = zeros(pop_size, D);
    fitness = benchmark(pos, I, 1);
    [best_fitness, best_idx] = min(fitness);
    best = pos(best_idx, :);

    % Record-keeping for evaluations and best fitness values
    tr = NaN(1, FE);
    tr(1:pop_size) = best_fitness;
    total_evals = pop_size;
    best_prev_eval = best_fitness;
    prob_table = ones(1, strategies) / strategies; % Equal initial probability for each strategy
    lambda = 0.75;  % Probability decay rate
    
    % Main loop for function evaluations
    iter = 0;
    while total_evals < FE
        iter = iter + 1; % Increment iteration counter
        frequency = Qmin + (Qmax - Qmin) * rand(pop_size, 1); % Random frequencies

        % Strategy selection for each bat
        for i = 1:pop_size
            % Choose a strategy based on probabilities
            cumulative_prob = cumsum(prob_table);
            strategy_idx = find(rand <= cumulative_prob); % Capture all possible matches
            if ~isempty(strategy_idx)
                selected_strategy = strategy_idx(1); % Force selection of the first strategy if multiple found
            else
                selected_strategy = 1; % Default to strategy 1 if no match (shouldn't happen but added as safeguard)
            end

            % Debugging output to verify selected strategy
          %  fprintf('Selected Strategy for bat %d: %d\n', i, selected_strategy);

            % Perform update based on selected strategy
            switch selected_strategy
                case 1  % Standard bat update
                    velocity(i, :) = velocity(i, :) + (pos(i, :) - best) .* frequency(i);
                    pos(i, :) = pos(i, :) + velocity(i, :);
                case 2  % Move toward worst bat for diversification
                    [~, worst_idx] = max(fitness);
                    worst = pos(worst_idx, :);
                    velocity(i, :) = velocity(i, :) + (pos(i, :) - worst) .* frequency(i);
                    pos(i, :) = pos(i, :) + velocity(i, :);
                case 3  % Levy flight for exploration
                    L = levy_flight(D); % Custom function for Levy flight
                    pos(i, :) = pos(i, :) + L .* (pos(i, :) - best);
                case 4  % Genetic crossover with best solution
                    pos(i, :) = 0.5 * (pos(i, :) + best);
                case 5  % PSO-inspired update
                    global_best = pos(best_idx, :);
                    velocity(i, :) = velocity(i, :) + rand * (global_best - pos(i, :)) + rand * (best - pos(i, :));
                    pos(i, :) = pos(i, :) + velocity(i, :);
                case 6  % Local disturbance
                    inertia = 0.9 - (0.5 * iter / FE);
                    pos(i, :) = pos(i, :) + inertia * randn(1, D);
                case 7  % Direct flight to the best
                    pos(i, :) = pos(i, :) + r0 * (best - pos(i, :));
                case 8  % Random walk around the best position
                    pos(i, :) = best + randn(1, D) * A;
            end

            % Ensure positions are within bounds after each strategy
            pos(i, :) = max(min(pos(i, :), range(2)), range(1));

            % Evaluate fitness of new position
            new_fitness = benchmark(pos(i, :), I, 1);
            total_evals = total_evals + 1;

            % Update position if fitness improves
            if new_fitness < fitness(i)
                fitness(i) = new_fitness;
                % Update probability table
                prob_table(selected_strategy) = prob_table(selected_strategy) * (1 + (1 - lambda));
            else
                prob_table(selected_strategy) = prob_table(selected_strategy) * lambda;
            end

            % Update the global best solution if improved
            if new_fitness < best_fitness
                best_fitness = new_fitness;
                best = pos(i, :);
            end
        end

        % Normalize probabilities
        prob_table = max(prob_table, 0.01);  % Ensure minimum probability for each strategy
        prob_table = prob_table / sum(prob_table);

        % Update loudness and pulse rate
        A = A * alpha;
        r = r0 * (1 - exp(-gamma * iter));

        % Early stopping based on relative improvement
        if useRelStop && total_evals >= evalWindow
            rel_improvement = abs(best_prev_eval - best_fitness) / max(1, abs(best_prev_eval));
            if rel_improvement < relTol
                tr(total_evals + 1:FE) = best_fitness;
                res = [best'; best_fitness];
                varargout{1} = 1:FE;
                varargout{2} = tr;
                varargout{3} = total_evals;
                return;
            end
            best_prev_eval = best_fitness;
        end

        % Record best fitness at each evaluation
        tr(total_evals - pop_size + 1:total_evals) = best_fitness;

        % Break if evaluation limit is reached
        if total_evals >= FE
            break;
        end
    end

    % Fill remaining records with best fitness
    if total_evals < FE
        tr(total_evals + 1:FE) = best_fitness;
    end

    % Output results
    res = [best'; best_fitness];
    varargout{1} = 1:FE;
    varargout{2} = tr;
    varargout{3} = total_evals;
end

% Custom Levy flight function for strategy 3
function L = levy_flight(dim)
    beta = 1.5;
    sigma = (gamma(1 + beta) * sin(pi * beta / 2) / (gamma((1 + beta) / 2) * beta * 2 ^ ((beta - 1) / 2)))^(1 / beta);
    u = randn(1, dim) * sigma;
    v = randn(1, dim);
    step = u ./ abs(v).^(1 / beta);
    L = step;
end
