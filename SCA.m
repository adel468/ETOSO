function [res, varargout] = SCA(I, D, range, FE, useRelStop, relTol, evalWindow)
    % Sine-Cosine Algorithm (SCA)

    % Initialize parameters
    pop_size = 30;                    % Population size
    a = 2;                             % Initial value of the parameter `a`
    functionIndex = I;                 % Benchmark function index
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
        r1 = a - (total_evals * (a / FE));  % Linearly decreasing parameter `a`

        % Update positions of each solution
        for i = 1:pop_size
            r2 = 2 * pi * rand();       % Random parameter in [0, 2*pi]
            r3 = 2 * rand();            % Random parameter in [0, 2]
            r4 = rand();                % Random parameter in [0, 1]

            % Update position based on sine or cosine
            if r4 < 0.5
                new_pos = pos(i, :) + r1 * sin(r2) * abs(r3 * best_pos - pos(i, :));
            else
                new_pos = pos(i, :) + r1 * cos(r2) * abs(r3 * best_pos - pos(i, :));
            end

            % Ensure the new position is within bounds
            new_pos = max(min(new_pos, VRmax), VRmin);

            % Evaluate the new position
            new_fitness = feval('benchmark', new_pos, functionIndex, 0);
            total_evals = total_evals + 1;

            % Update position and fitness if improvement is achieved
            if new_fitness < fitness(i)
                pos(i, :) = new_pos;
                fitness(i) = new_fitness;

                % Update global best if this position is the new best
                if new_fitness < best_fitness
                    best_fitness = new_fitness;
                    best_pos = new_pos;
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
