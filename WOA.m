function [res, varargout] = WOA(I, D, range, FE, useRelStop, relTol, evalWindow)
    % Whale Optimization Algorithm (WOA)

    % Initialize parameters
    pop_size = 30;                    % Population size
    functionIndex = I;                % Benchmark function index
    VRmin = range(1);
    VRmax = range(2);

    % Initialize whale population positions and evaluate initial fitness
    pos = VRmin + rand(pop_size, D) * (VRmax - VRmin);
    fitness = feval('benchmark', pos, functionIndex, 1);

    % Initialize best fitness tracking
    [best_fitness, best_idx] = min(fitness);
    best_pos = pos(best_idx, :);

    % Initialize record-keeping for function evaluations and best values
    tr = NaN(1, FE);                  % Track best fitness after each evaluation
    tr(1:pop_size) = best_fitness;
    total_evals = pop_size;
    best_prev_eval = best_fitness;    % Track previous best for relative improvement

    % Main loop for function evaluations
    while total_evals < FE
        % Parameter `a` decreases over time
        a = 2 - total_evals * (2 / FE);

        for i = 1:pop_size
            r = rand();  % Random number for determining movement mechanism

            % Position update based on WOA mechanisms
            if r < 0.5
                % Shrinking encircling mechanism
                A = 2 * a * rand() - a;
                C = 2 * rand();
                D = abs(C * best_pos - pos(i, :));
                new_pos = best_pos - A * D;
            else
                % Spiral updating position mechanism
                b = 1;                     % Constant defining logarithmic spiral
                l = -1 + 2 * rand();       % Random number in [-1,1]
                distance_to_best = abs(best_pos - pos(i, :));
                new_pos = distance_to_best * exp(b * l) * cos(l * 2 * pi) + best_pos;
            end

            % Ensure the new position is within bounds
            pos(i, :) = max(min(new_pos, VRmax), VRmin);
            new_fitness = feval('benchmark', pos(i, :), functionIndex, 0);
            total_evals = total_evals + 1;

            % Update fitness if the new position is better
            if new_fitness < fitness(i)
                fitness(i) = new_fitness;

                % Update global best if this position is the new best
                if new_fitness < best_fitness
                    best_fitness = new_fitness;
                    best_pos = pos(i, :);
                end
            end

            % Store best fitness after each evaluation if within bounds of FE
            if total_evals <= FE
                tr(total_evals) = best_fitness;
            end

            % Early stopping based on relative improvement
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

            % Stop if FE limit reached
            if total_evals >= FE
                break;
            end
        end
        if total_evals >= FE
            break;
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
