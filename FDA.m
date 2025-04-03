function [res, varargout] = FDA(I, D, range, FE, useRelStop, relTol, evalWindow)
    % Flow Direction Algorithm (FDA) Implementation

    fncton = I;           % Function index
    pop_size = 30;        % Population size (number of flows)
    alpha = 25;           % Randomness parameter for initial position
    beta = 3;             % Number of neighbors
    gamma = 1;            % Absorption coefficient (influence factor)
    VRmin = range(1);
    VRmax = range(2);

    % Initialize flow positions and evaluate initial fitness
    pos = VRmin + (rand(pop_size, D) + alpha * randn(pop_size, D)) .* (VRmax - VRmin);
    fitness = benchmark(pos, fncton, 1);

    % Initialize best fitness tracking
    [best_fitness, best_idx] = min(fitness);
    best_pos = pos(best_idx, :);

    % Initialize record-keeping for function evaluations and best values
    tr = NaN(1, FE);                % Track best fitness after each evaluation
    tr(1:pop_size) = best_fitness;
    total_evals = pop_size;
    best_prev_eval = best_fitness;   % Track previous best for relative improvement

    % Main loop for function evaluations
    while total_evals < FE
        % Linearly decrease neighborhood radius `A`
        A = 10 * (1 - total_evals / FE);  % Decreases from 10 to 0 over evaluations

        % Generate and evaluate neighbors for each flow
        for i = 1:pop_size
            for j = 1:beta
                % Generate neighbor position and ensure bounds
                neighbor_pos = pos(i, :) + randn(1, D) .* A;
                neighbor_pos = min(max(neighbor_pos, VRmin), VRmax);

                % Evaluate neighbor fitness
                neighbor_fitness = benchmark(neighbor_pos, fncton, 0);
                total_evals = total_evals + 1;

                % Update tr continuously for each evaluation
                if total_evals <= FE
                    tr(total_evals) = best_fitness;
                end

                % Update the position if the neighbor is better
                if neighbor_fitness < fitness(i)
                    pos(i, :) = neighbor_pos;
                    fitness(i) = neighbor_fitness;
                    if neighbor_fitness < best_fitness
                        best_fitness = neighbor_fitness;
                        best_pos = neighbor_pos;
                    end
                end
            end
        end

        % Update flow positions based on best neighbors
        new_pos = pos;
        for i = 1:pop_size
            % Find the best neighbor among evaluated ones
            [~, best_neighbor_idx] = min(fitness);
            best_neighbor = pos(best_neighbor_idx, :);

            % Move towards best neighbor if it's better, otherwise random movement
            if fitness(i) > fitness(best_neighbor_idx)
                new_pos(i, :) = pos(i, :) + gamma * (best_neighbor - pos(i, :));
            else
                new_pos(i, :) = pos(i, :) + randn(1, D) .* A;  % Random movement
            end

            % Ensure new position is within bounds
            new_pos(i, :) = min(max(new_pos(i, :), VRmin), VRmax);

            % Evaluate new position and update if better
            new_fitness = benchmark(new_pos(i, :), fncton, 0);
            total_evals = total_evals + 1;

            % Track best fitness after each evaluation
            if total_evals <= FE
                tr(total_evals) = best_fitness;
            end

            if new_fitness < fitness(i)
                pos(i, :) = new_pos(i, :);
                fitness(i) = new_fitness;
                if new_fitness < best_fitness
                    best_fitness = new_fitness;
                    best_pos = new_pos(i, :);
                end
            end

            % Early stopping check based on relative improvement
            if useRelStop && total_evals >= evalWindow
                rel_improvement = abs(best_prev_eval - best_fitness) / max(1, abs(best_prev_eval));
                if rel_improvement < relTol
                    tr(total_evals + 1:FE) = best_fitness;  % Fill remaining entries
                    res = [best_pos'; best_fitness];
                    varargout{1} = 1:FE;
                    varargout{2} = tr;
                    varargout{3} = total_evals;
                    return;
                end
                best_prev_eval = best_fitness;
            end

            % Stop if total evaluations reach FE
            if total_evals >= FE
                break;
            end
        end
    end

    % Fill remaining `tr` entries if FE limit is reached without early stopping
    if total_evals < FE
        tr(total_evals + 1:FE) = best_fitness;
    end

    % Final output
    res = [best_pos'; best_fitness];
    varargout{1} = 1:FE;
    varargout{2} = tr;
    varargout{3} = total_evals;
end
