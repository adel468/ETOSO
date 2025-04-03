function [res, varargout] = SMA(I, D, range, FE, useRelStop, relTol, evalWindow)
    % Slime Mould Algorithm (SMA)

    % Initialize parameters
    pop_size = 30;                    % Population size
    vaMax = 1;                        % Maximum value for 'va'
    vaMin = 0.01;                     % Minimum value for 'va'
    functionIndex = I;                % Benchmark function index
    VRmin = range(1);
    VRmax = range(2);

    % Initialize population positions and evaluate initial fitness
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
        va = vaMax - ((vaMax - vaMin) * (total_evals / FE)); % Update 'va' over evaluations

        % Update positions of each slime mould
        for i = 1:pop_size
            VC = 1 / (1 + exp(-fitness(i) + max(fitness))); % Sigmoid-based scaling of fitness
            p = va * VC;                                    % Probability factor

            % Update position based on SMA behavior
            if rand() < p
                % Move towards the best position
                pos(i, :) = pos(i, :) + (best_pos - pos(i, :));
            else
                % Conduct random exploration
                random_mould = randi(pop_size);
                direction = (pos(random_mould, :) - pos(i, :)) + randn(1, D) * 0.1; % Add noise for exploration
                pos(i, :) = pos(i, :) + direction;
            end

            % Enforce bounds on the new position
            pos(i, :) = max(min(pos(i, :), VRmax), VRmin);

            % Evaluate the new position
            new_fitness = feval('benchmark', pos(i, :), functionIndex, 0);
            total_evals = total_evals + 1;

            % Update position and fitness if improvement is achieved
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

            % Stop if total evaluations reach FE
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
