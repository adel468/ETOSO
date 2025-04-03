function [res, varargout] = TLBO(I, D, range, FE, useRelStop, relTol, evalWindow)
    % Teaching-Learning-Based Optimization (TLBO)

    % Initialize parameters
    pop_size = 30;                    % Population size
    functionIndex = I;                % Benchmark function index
    VRmin = range(1);
    VRmax = range(2);

    % Initialize learner population positions and evaluate initial fitness
    pos = VRmin + rand(pop_size, D) * (VRmax - VRmin);
    fitness = feval('benchmark', pos, functionIndex, 1);

    % Initialize best fitness tracking
    [best_fitness, best_idx] = min(fitness);
    best_pos = pos(best_idx, :);

    % Initialize record-keeping for function evaluations and best values
    tr = NaN(1, FE);                   % Track best fitness after each evaluation
    tr(1:pop_size) = best_fitness;
    total_evals = pop_size;
    best_prev_eval = best_fitness;     % Track previous best for relative improvement

    % Main loop for function evaluations
    while total_evals < FE
        % Teacher Phase
        mean_pos = mean(pos, 1);
        for i = 1:pop_size
            new_pos = pos(i, :) + rand(1, D) .* (best_pos - round(rand * mean_pos));
            new_pos = max(min(new_pos, VRmax), VRmin); % Ensure within bounds
            new_fitness = feval('benchmark', new_pos, functionIndex, 0);
            total_evals = total_evals + 1;

            % Update position if new position is better
            if new_fitness < fitness(i)
                fitness(i) = new_fitness;
                pos(i, :) = new_pos;

                % Update global best if new fitness is the best found
                if new_fitness < best_fitness
                    best_fitness = new_fitness;
                    best_pos = new_pos;
                end
            end

            % Track best fitness if within FE bounds
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

        % Learner Phase
        for i = 1:pop_size
            j = randi(pop_size);
            while j == i
                j = randi(pop_size);
            end

            % Update position based on learning interaction
            if fitness(j) < fitness(i)
                new_pos = pos(i, :) + rand(1, D) .* (pos(j, :) - pos(i, :));
            else
                new_pos = pos(i, :) + rand(1, D) .* (pos(i, :) - pos(j, :));
            end
            new_pos = max(min(new_pos, VRmax), VRmin);
            new_fitness = feval('benchmark', new_pos, functionIndex, 0);
            total_evals = total_evals + 1;

            % Update if new fitness is better
            if new_fitness < fitness(i)
                fitness(i) = new_fitness;
                pos(i, :) = new_pos;

                % Update global best if new fitness is the best found
                if new_fitness < best_fitness
                    best_fitness = new_fitness;
                    best_pos = new_pos;
                end
            end

            % Track best fitness if within FE bounds
            if total_evals <= FE
                tr(total_evals) = best_fitness;
            end

            % Early stopping based on relative improvement
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
