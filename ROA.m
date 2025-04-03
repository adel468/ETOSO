function [res, varargout] = ROA(I, D, range, FE, useRelStop, relTol, evalWindow)
    % Remora Optimization Algorithm (ROA) based on the original papers

    % Algorithm Constants
    ps = 30;                  % Population size
    Att = 0.9;                % Initial attachment parameter for exploration
    Att_end = 0.1;            % Final attachment parameter for exploration
    SP = 0.499;               % Switching probability for host selection (SFO or WOA)
    VRmin = range(1);         % Minimum bound of search space
    VRmax = range(2);         % Maximum bound of search space
    functionIndex = I;        % Benchmark function index

    % Initialize remora population and evaluate initial fitness
    pos = VRmin + rand(ps, D) * (VRmax - VRmin);
    fitness = feval('benchmark', pos, functionIndex, 1);

    % Initialize best fitness tracking
    [best_fitness, best_idx] = min(fitness);
    best_pos = pos(best_idx, :);

    % Initialize record-keeping for function evaluations and best values
    tr = NaN(1, FE);               % Track best fitness after each evaluation
    tr(1:ps) = best_fitness;
    total_evals = ps;
    best_prev_eval = best_fitness; % Track previous best for relative improvement

    % Main loop for function evaluations
    while total_evals < FE
        % Update attachment factor linearly over evaluations
        attachment_factor = Att - (Att - Att_end) * (total_evals / FE);

        % Update positions of each remora
        for i = 1:ps
            % Determine host selection mechanism
            if rand() < SP
                % Exploitation phase (WOA mechanism)
                D_host = abs(best_pos - pos(i, :));  % Distance to best position
                l = (2 * rand() - 1);                % Random scalar factor
                b = 1.5;                             % WOA constant for logarithmic spiral

                if rand() < 0.5
                    % Shrinking encircling mechanism
                    pos(i, :) = best_pos - attachment_factor * D_host;
                else
                    % Spiral update mechanism
                    pos(i, :) = D_host * exp(b * l) .* cos(2 * pi * l) + best_pos;
                end
            else
                % Exploration phase (SFO mechanism)
                random_remora = randi(ps);  % Choose a random remora
                pos(i, :) = best_pos - attachment_factor * (best_pos + pos(random_remora, :)) / 2 - pos(random_remora, :);
            end

            % Ensure new position is within bounds
            pos(i, :) = max(min(pos(i, :), VRmax), VRmin);

            % Evaluate the new position and update fitness
            new_fitness = feval('benchmark', pos(i, :), functionIndex, 0);
            total_evals = total_evals + 1;

            % Update global best if the new position is better
            if new_fitness < fitness(i)
                fitness(i) = new_fitness;
                if new_fitness < best_fitness
                    best_fitness = new_fitness;
                    best_pos = pos(i, :);
                end
            end

            % Store best fitness after each evaluation within bounds of FE
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
