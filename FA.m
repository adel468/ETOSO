function [res, varargout] = FA(I, D, range, FE, useRelStop, relTol, evalWindow)
   % Firefly Algorithm (FA) Implementation

    fncton = I;           % Function index
    ps = 30;              % Population size (number of fireflies)
    alpha = 1;          % Randomness parameter
    gamma = 1;            % Light absorption coefficient
    beta0 = 0.2;          % Base attractiveness
    VRmin = range(1);
    VRmax = range(2);

    % Initialize firefly population and evaluate initial fitness
    pos = VRmin + rand(ps, D) * (VRmax - VRmin);
    fitness = benchmark(pos, fncton, 1);

    % Initialize best fitness tracking
    [best_fitness, best_idx] = min(fitness);
    best_pos = pos(best_idx, :);

    % Initialize record-keeping for function evaluations and best values
    tr = NaN(1, FE);                % Track best fitness after each evaluation
    tr(1:ps) = best_fitness;         % Initial fitness values
    total_evals = ps;
    best_prev_eval = best_fitness;   % Track previous best for relative improvement

    % Main loop for function evaluations
    while total_evals < FE
        % Compare fireflies and update positions
        new_pos = pos;
        for i = 1:ps
            for j = 1:ps
                if fitness(i) > fitness(j)  % Move firefly i towards brighter j
                    r = norm(pos(i, :) - pos(j, :), 2);  % Euclidean distance
                    beta = beta0 * exp(-gamma * r^2);    % Attractiveness

                    % Update position with attraction and randomness
                    new_pos(i, :) = new_pos(i, :) + beta * (pos(j, :) - pos(i, :)) + ...
                                    alpha * (rand(1, D) - 0.5);

                    % Bound checking
                    new_pos(i, :) = max(min(new_pos(i, :), VRmax), VRmin);
                end
            end

            % Evaluate new position and update if better
            new_fitness = benchmark(new_pos(i, :), fncton, 0);
            total_evals = total_evals + 1;

            % Update position and fitness if the new solution is better
            if new_fitness < fitness(i)
                pos(i, :) = new_pos(i, :);
                fitness(i) = new_fitness;
                if new_fitness < best_fitness
                    best_fitness = new_fitness;
                    best_pos = new_pos(i, :);
                end
            end

            % Track best fitness after each evaluation within bounds of FE
            if total_evals <= FE
                tr(total_evals) = best_fitness;
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
