function [res, varargout] = BOA(I, D, range, FE, useRelStop, relTol, evalWindow)
    % Butterfly Optimization Algorithm (BOA)
    fncton = I;  % Function index

    % Initialize parameters
    ps = 30;       % Population size (Number of butterflies)
    c = 0.01;      % Sensory modality (Smell intensity)
    a = 0.1;       % Power exponent
    p = 0.8;       % Probability switch
    VRmin = repmat(range(1), 1, D); % Minimum value range
    VRmax = repmat(range(2), 1, D); % Maximum value range

    % Initialize butterfly population
    pos = rand(ps, D) .* (VRmax - VRmin) + VRmin;
    fitness = benchmark(pos, fncton, 1);  % Initial fitness evaluation
    [best_fitness, best_idx] = min(fitness);
    best_pos = pos(best_idx, :);

    % Initialize record-keeping for function evaluations and best values
    tr = NaN(1, FE);           % Store best fitness after each evaluation
    tr(1:ps) = best_fitness;    % Initial fitness values
    total_evals = ps;           % Start evaluation counter
    best_prev_eval = best_fitness;  % Track the previous best fitness for relative improvement

    % Main loop for function evaluations
    while total_evals < FE
        for i = 1:ps
            % Calculate fragrance (proportional to fitness)
            fragrance = c * fitness(i)^a;

            % Determine movement type (global or local)
            if rand() < p
                % Global search (towards the best solution)
                candidate = best_pos + fragrance * (rand(1, D) - 0.5) .* (VRmax - VRmin);
            else
                % Local search (towards a random butterfly)
                rand_idx = randi(ps);
                while rand_idx == i  % Avoid moving towards oneself
                    rand_idx = randi(ps);
                end
                candidate = pos(rand_idx, :) + fragrance * (rand(1, D) - 0.5) .* (VRmax - VRmin);
            end

            % Bound checking
            candidate = max(min(candidate, VRmax), VRmin);

            % Evaluate new position
            candidate_fitness = benchmark(candidate, fncton, 0);
            total_evals = total_evals + 1;

            % Update if a better solution is found
            if candidate_fitness < fitness(i)
                pos(i, :) = candidate;
                fitness(i) = candidate_fitness;
                if candidate_fitness < best_fitness
                    best_fitness = candidate_fitness;
                    best_pos = candidate;
                end
            end

            % Record the best fitness for each evaluation
            if total_evals <= FE
                tr(total_evals) = best_fitness;
            end

            % Early stopping check
            if useRelStop && total_evals >= evalWindow
                rel_improvement = abs(best_prev_eval - best_fitness) / max(1, abs(best_prev_eval));
                if rel_improvement < relTol
                    tr(total_evals + 1:FE) = best_fitness;
                    res = [best_pos'; best_fitness];
                    varargout{1} = 1:FE;
                    varargout{2} = tr;
                    varargout{3} = total_evals;
                    return;
                end
                best_prev_eval = best_fitness;
            end

            % Check if FE evaluations have been reached
            if total_evals >= FE
                break;
            end
        end
    end

    % Fill remaining `tr` values if FE limit is reached without early stopping
    if total_evals < FE
        tr(total_evals + 1:FE) = best_fitness;
    end

    % Final output
    res = [best_pos'; best_fitness];
    varargout{1} = 1:FE;
    varargout{2} = tr;
    varargout{3} = total_evals;
end
