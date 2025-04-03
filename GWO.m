function [res, varargout] = GWO(I, D, range, FE, useRelStop, relTol, evalWindow)
    % Grey Wolf Optimizer (GWO)

    % Initialize parameters
    pop_size = 30;                    % Population size
    functionIndex = I;                % Benchmark function index
    VRmin = range(1);
    VRmax = range(2);

    % Initialize grey wolf pack and evaluate initial fitness
    pos = VRmin + rand(pop_size, D) * (VRmax - VRmin);
    fitness = feval('benchmark', pos, functionIndex, 1);

    % Determine initial alpha, beta, and delta wolves
    [sorted_fitness, sorted_indices] = sort(fitness);
    alpha = pos(sorted_indices(1), :);
    beta = pos(sorted_indices(2), :);
    delta = pos(sorted_indices(3), :);
    best_fitness = sorted_fitness(1);

    % Initialize record-keeping for function evaluations and best values
    tr = NaN(1, FE);                  % Track best fitness after each evaluation
    tr(1:pop_size) = best_fitness;
    total_evals = pop_size;
    best_prev_eval = best_fitness;    % Track previous best for relative improvement

    % Main loop for function evaluations
    while total_evals < FE
        % Update parameter `a` linearly
        a = 2 * (1 - total_evals / FE);

        % Update positions of each grey wolf
        new_pos = pos;
        for i = 1:pop_size
            % Calculate position updates relative to alpha, beta, and delta
            for j = 1:D
                r1 = rand; r2 = rand;
                A1 = 2 * a * r1 - a;
                C1 = 2 * r2;

                D_alpha = abs(C1 * alpha(j) - pos(i, j));
                X1 = alpha(j) - A1 * D_alpha;

                r1 = rand; r2 = rand;
                A2 = 2 * a * r1 - a;
                C2 = 2 * r2;

                D_beta = abs(C2 * beta(j) - pos(i, j));
                X2 = beta(j) - A2 * D_beta;

                r1 = rand; r2 = rand;
                A3 = 2 * a * r1 - a;
                C3 = 2 * r2;

                D_delta = abs(C3 * delta(j) - pos(i, j));
                X3 = delta(j) - A3 * D_delta;

                % New position calculation
                new_pos(i, j) = (X1 + X2 + X3) / 3;
            end

            % Ensure the new position is within bounds
            new_pos(i, :) = max(min(new_pos(i, :), VRmax), VRmin);
        end

        % Evaluate new positions
        new_fitness = feval('benchmark', new_pos, functionIndex, 0);
        total_evals = total_evals + pop_size;

        % Update population positions and fitness
        pos = new_pos;
        fitness = new_fitness;

        % Update alpha, beta, and delta based on new fitness
        [sorted_fitness, sorted_indices] = sort(fitness);
        if sorted_fitness(1) < best_fitness
            alpha = pos(sorted_indices(1), :);
            best_fitness = sorted_fitness(1);
        end
        beta = pos(sorted_indices(2), :);
        delta = pos(sorted_indices(3), :);

        % Track best fitness only if it improves
        for j = total_evals - pop_size + 1:total_evals
            if j <= FE
                tr(j) = best_fitness;
            end
        end

        % Early stopping check based on relative improvement
        if useRelStop && total_evals >= evalWindow
            rel_improvement = abs(best_prev_eval - best_fitness) / max(1, abs(best_prev_eval));
            if rel_improvement < relTol
                tr(total_evals + 1:FE) = best_fitness;  % Fill remaining entries
                res = [alpha'; best_fitness];
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

    % Fill remaining `tr` entries if FE limit is reached without early stopping
    if total_evals < FE
        tr(total_evals + 1:FE) = best_fitness;
    end

    % Final output
    res = [alpha'; best_fitness];
    varargout{1} = 1:FE;
    varargout{2} = tr;
    varargout{3} = total_evals;
end
