function [res, varargout] = SSA(I, D, range, FE, useRelStop, relTol, evalWindow)
    % Orthogonal Opposition-Based Salp Swarm Algorithm (OOSSA)

    % Initialize parameters
    pop_size = 30;                    % Population size as per paper
    functionIndex = I;                % Benchmark function index
    VRmin = range(1);
    VRmax = range(2);

    % Initialize salp population positions and evaluate initial fitness
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
        % Coefficient `c` as per paper (exponential decay)
        c = 2 * exp(-(4 * total_evals / FE)^2);

        % Sort fitness and update best salp
        [fitness, sorted_indices] = sort(fitness);
        pos = pos(sorted_indices, :);  % Sort positions by fitness
        best_pos = pos(1, :);          % Best position

        % Adaptive leader-follower mechanism
        num_leaders = max(1, round(pop_size * (1 - total_evals / FE))); % Adaptive leader count

        for i = 1:pop_size
            if i <= num_leaders
                % Leader salp position update based on `c` as per OOSSA
                for j = 1:D
                    if rand < 0.5
                        pos(i, j) = best_pos(j) + c * ((VRmax - VRmin) * rand + VRmin);
                    else
                        pos(i, j) = best_pos(j) - c * ((VRmax - VRmin) * rand + VRmin);
                    end
                end
            else
                % Follower salp position update using average with prior salp
                pos(i, :) = (pos(i, :) + pos(i - 1, :)) / 2;
            end

            % Orthogonal Lens Opposition-Based Learning (OLOBL) as per paper
            olobl_pos = VRmin + VRmax - pos(i, :) + rand * (VRmax - VRmin) .* sign(rand(1, D) - 0.5);
            olobl_fitness = feval('benchmark', olobl_pos, functionIndex, 0);

            % Update to OLOBL position if better
            if olobl_fitness < fitness(i)
                pos(i, :) = olobl_pos;
                fitness(i) = olobl_fitness;
            end

            % Ensure positions are within bounds
            pos(i, :) = max(min(pos(i, :), VRmax), VRmin);

            % Evaluate the new position
            new_fitness = feval('benchmark', pos(i, :), functionIndex, 0);
            total_evals = total_evals + 1;

            % Update fitness if new position is better
            if new_fitness < fitness(i)
                fitness(i) = new_fitness;

                % Update global best if new best found
                if new_fitness < best_fitness
                    best_fitness = new_fitness;
                    best_pos = pos(i, :);
                end
            end

            % Record best fitness after each evaluation within bounds
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
