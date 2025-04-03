function [res, varargout] = MKA(I, D, range, FE, useRelStop, relTol, evalWindow)
    % Monkey Algorithm Parameters
    pop_size = 10;                      % Increased Population Size
    step_length = 1;                 % Smaller Step Length for Climbing Precision
    climb_number = 10;                  % Number of climb iterations (Nc)
    eyesight = 0.1;                     % Maximum watch-jump distance (b)
    somersault_interval = [-1, 1];      % Somersault range for exploration [c, d]
%    FE = 6 * FE / climb_number;

    % Initialize monkey population and evaluate initial fitness
    pos = range(1) + rand(pop_size, D) * (range(2) - range(1));
    fitness = benchmark(pos, I, 1);

    % Initialize best fitness tracking
    [best_fitness, best_idx] = min(fitness);
    best = pos(best_idx, :);

    % Initialize record-keeping for function evaluations and best values
    tr = NaN(1, FE);                    % Track best fitness after each evaluation
    tr(1:pop_size) = best_fitness;
    total_evals = pop_size;
    best_prev_eval = best_fitness;      % Track previous best for relative improvement

    % Main loop for function evaluations
    while total_evals < FE
        % Climbing Phase
        for climb_iter = 1:climb_number
            for i = 1:pop_size
                % Generate new climb position
                climb_pos = pos(i, :) + step_length * randn(1, D);
                climb_pos = max(min(climb_pos, range(2)), range(1));  % Bound checking

                % Evaluate the climb position
                climb_fitness = benchmark(climb_pos, I, 0);
                total_evals = total_evals + 1;

                % Update position and fitness if improvement
                if climb_fitness < fitness(i)
                    pos(i, :) = climb_pos;
                    fitness(i) = climb_fitness;
                    % Update global best if this position is new best
                    if climb_fitness < best_fitness
                        best_fitness = climb_fitness;
                        best = climb_pos;
                    end
                end

                % Store the best fitness after each evaluation if within bounds of FE
                if total_evals <= FE
                    tr(total_evals) = best_fitness;
                end

                % Early stopping check based on relative improvement
                if useRelStop && total_evals >= evalWindow
                    rel_improvement = abs(best_prev_eval - best_fitness) / max(1, abs(best_prev_eval));
                    if rel_improvement < relTol
                        tr(total_evals + 1:FE) = best_fitness;
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
            if total_evals >= FE
                break;
            end
        end

        % Watch-Jump Phase
        for i = 1:pop_size
            y = pos(i, :) + (rand(1, D) * 2 - 1) * eyesight;
            y = max(min(y, range(2)), range(1));  % Bound checking

            % Evaluate the new position
            y_fitness = benchmark(y, I, 0);
            total_evals = total_evals + 1;

            % Update if the new position is better
            if y_fitness < fitness(i)
                pos(i, :) = y;
                fitness(i) = y_fitness;
                if y_fitness < best_fitness
                    best_fitness = y_fitness;
                    best = y;
                end
            end

            % Store the best fitness if within bounds of FE
            if total_evals <= FE
                tr(total_evals) = best_fitness;
            end

            % Early stopping based on relative improvement
            if useRelStop && total_evals >= evalWindow
                rel_improvement = abs(best_prev_eval - best_fitness) / max(1, abs(best_prev_eval));
                if rel_improvement < relTol
                    tr(total_evals + 1:FE) = best_fitness;
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
        if total_evals >= FE
            break;
        end

        % Somersault Phase
        pivot = mean(pos);  % Calculate the barycenter
        for i = 1:pop_size
            alpha = rand * (somersault_interval(2) - somersault_interval(1)) + somersault_interval(1);
            y = pos(i, :) + alpha * (pivot - pos(i, :));
            y = max(min(y, range(2)), range(1));  % Bound checking

            % Evaluate the somersault position
            somersault_fitness = benchmark(y, I, 0);
            total_evals = total_evals + 1;

            % Update if the new position is better
            if somersault_fitness < fitness(i)
                pos(i, :) = y;
                fitness(i) = somersault_fitness;
                if somersault_fitness < best_fitness
                    best_fitness = somersault_fitness;
                    best = y;
                end
            end

            % Store best fitness if within bounds of FE
            if total_evals <= FE
                tr(total_evals) = best_fitness;
            end

            % Early stopping based on relative improvement
            if useRelStop && total_evals >= evalWindow
                rel_improvement = abs(best_prev_eval - best_fitness) / max(1, abs(best_prev_eval));
                if rel_improvement < relTol
                    tr(total_evals + 1:FE) = best_fitness;
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
        if total_evals >= FE
            break;
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
