function [res, varargout] = PDO(I, D, range, FE, useRelStop, relTol, evalWindow)
    % Modified Prairie Dog Optimization (MPDO)
    
    % Initialize parameters
    pop_size = 30;              % Population size
    functionIndex = I;           % Benchmark function index
    rho = 0.1;                   % Scaling factor for migration
    delta = 0.005;               % Adjustment factor updating position

    VRmin = range(1);
    VRmax = range(2);

    % Initialize prairie dog population using chaotic tent initialization
    pos = chaoticTentInitialization(pop_size, D, VRmin, VRmax);
    fitness = feval('benchmark', pos, functionIndex, 1);

    % Initialize best fitness tracking
    [best_fitness, best_idx] = min(fitness);
    best_pos = pos(best_idx, :);

    % Initialize record-keeping for function evaluations and best values
    tr = NaN(1, FE);                % Track best fitness after each evaluation
    tr(1:pop_size) = best_fitness;
    total_evals = pop_size;
    best_prev_eval = best_fitness;  % Track previous best for relative improvement

    % Main loop for function evaluations
    while total_evals < FE
        for i = 1:pop_size
            % Generate opposition-based learning solution
            X_opposing = VRmax + VRmin - pos(i, :);
            pos(i, :) = (pos(i, :) + X_opposing) / 2;  % Update with opposition-based learning

            % Update prairie dog position with migration
            migration_factor = rho + (1 - rho) * rand();
            if rand() < 0.5
                % Digging/utilizing position
                random_dog = randi(pop_size);
                pos(i, :) = pos(random_dog, :) - migration_factor * (pos(random_dog, :) - pos(i, :));
            else
                % Utilizing the best position
                pos(i, :) = best_pos - migration_factor * (best_pos - pos(i, :)) + delta * (VRmax - VRmin) .* rand(1, D);
            end

            % Ensure the new position is within bounds
            pos(i, :) = max(min(pos(i, :), VRmax), VRmin);

            % Evaluate the new position
            fitness(i) = feval('benchmark', pos(i, :), functionIndex, 0);
            total_evals = total_evals + 1;

            % Update global best if the new position is better
            if fitness(i) < best_fitness
                best_fitness = fitness(i);
                best_pos = pos(i, :);
            end

            % Store the best fitness after each evaluation if within bounds of FE
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

function pos = chaoticTentInitialization(ps, D, VRmin, VRmax)
    % Chaotic Tent Initialization for MPDO
    pos = zeros(ps, D);

    for i = 1:ps
        % Generate a random chaotic number
        x = rand();  % Random initial value in [0, 1]

        % Apply the tent map to generate chaotic positions
        for j = 1:D
            if x < 0.5
                x = 2 * x;
            else
                x = 2 * (1 - x);
            end
            % Scale the chaotic number to the defined range
            pos(i, j) = VRmin + x * (VRmax - VRmin);
        end
    end
end
