function [res, varargout] = GSA(I, D, range, FE, useRelStop, relTol, evalWindow)
    % Gravitational Search Algorithm (GSA)
    
    % Initialize parameters
    pop_size = 30;            % Population size
    G0 = 100;                 % Initial gravitational constant
    alpha = 20;               % Constant for decay rate

    % Initialize population and evaluate initial fitness
    pos = range(1) + rand(pop_size, D) * (range(2) - range(1));
    fitness = benchmark(pos, I, 1);

    % Initialize best fitness tracking
    [best_fitness, best_idx] = min(fitness);
    best = pos(best_idx, :);

    % Initialize record-keeping for function evaluations and best values
    tr = NaN(1, FE);                % Track best fitness after each evaluation
    tr(1:pop_size) = best_fitness;
    total_evals = pop_size;
    best_prev_eval = best_fitness;  % Track previous best for relative improvement

    % Main loop for function evaluations
    while total_evals < FE
        % Update gravitational constant `G`
        G = G0 * exp(-alpha * total_evals / FE);

        % Calculate the mass for each agent based on fitness
        fitness_shifted = fitness - min(fitness) + 1e-8;  % Shift to avoid division by zero
        total_mass = sum(1 ./ fitness_shifted);           % Total mass
        mass = (1 ./ fitness_shifted) / total_mass;       % Mass for each agent

        % Update positions of each agent
        new_pos = pos;
        for i = 1:pop_size
            % Gravitational force accumulation
            acceleration = zeros(1, D);
            for j = 1:pop_size
                if i ~= j
                    r = norm(pos(i, :) - pos(j, :)) + 1e-8;  % Calculate distance, avoiding division by zero
                    direction = (pos(j, :) - pos(i, :)) / r;
                    acceleration = acceleration + G * mass(j) * direction / r^2;
                end
            end
            
            % Update position with randomization
            new_pos(i, :) = pos(i, :) + rand(1, D) .* acceleration;
            new_pos(i, :) = max(min(new_pos(i, :), range(2)), range(1));  % Bound checking

            % Evaluate new position and update if better
            new_fitness = benchmark(new_pos(i, :), I, 1);
            total_evals = total_evals + 1;

            if new_fitness < fitness(i)
                pos(i, :) = new_pos(i, :);
                fitness(i) = new_fitness;
                if new_fitness < best_fitness
                    best_fitness = new_fitness;
                    best = new_pos(i, :);
                end
            end

            % Track best fitness after each evaluation if within bounds of FE
            if total_evals <= FE
                tr(total_evals) = best_fitness;
            end

            % Early stopping check based on relative improvement
            if useRelStop && total_evals >= evalWindow
                rel_improvement = abs(best_prev_eval - best_fitness) / max(1, abs(best_prev_eval));
                if rel_improvement < relTol
                    tr(total_evals + 1:FE) = best_fitness;  % Fill remaining entries
                    res = [best'; best_fitness];
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
    res = [best'; best_fitness];
    varargout{1} = 1:FE;
    varargout{2} = tr;
    varargout{3} = total_evals;
end
