function [res, varargout] = HHO(I, D, range, FE, useRelStop, relTol, evalWindow)
    % Harris Hawks Optimization (HHO)

    % Initialize parameters
    ps = 30;             % Population size
    sfn = I;             % Function index
    VRmin = range(1);
    VRmax = range(2);

    % Initialize hawk population and evaluate initial fitness
    pos = VRmin + rand(ps, D) * (VRmax - VRmin);
    out = feval('benchmark', pos, sfn, 1);

    % Determine initial best position
    [bestval, sorted_indices] = min(out);
    best_pos = pos(sorted_indices, :);

    % Initialize record-keeping for function evaluations and best values
    tr = NaN(1, FE);                % Track best fitness after each evaluation
    tr(1:ps) = bestval;
    total_evals = ps;
    best_prev_eval = bestval;       % Track previous best for relative improvement

    % Main loop for function evaluations
    while total_evals < FE
        E0 = 2 * rand() - 1;                    % Initial energy
        E = 2 * E0 * (1 - total_evals / FE);    % Decreasing energy

        % Update positions of each hawk
        new_pos = pos;
        for i = 1:ps
            r = rand();
            if abs(E) >= 1
                % Exploration phase
                if r < 0.5
                    % Random exploration
                    rand_hawk = randi(ps);
                    new_pos(i, :) = pos(rand_hawk, :) - rand() * abs(pos(rand_hawk, :) - 2 * rand() * pos(i, :));
                else
                    % Exploiting the best
                    new_pos(i, :) = best_pos - rand() * abs(best_pos - pos(i, :));
                end
            else
                % Exploitation phase
                q = rand();
                if q < 0.5 && abs(E) < 0.5
                    % Soft besiege
                    new_pos(i, :) = best_pos - E * abs(best_pos - pos(i, :));
                elseif q >= 0.5 && abs(E) >= 0.5
                    % Hard besiege
                    new_pos(i, :) = best_pos - E * abs(best_pos - mean(pos));
                else
                    % Rapid dives
                    X_rand = VRmin + rand(1, D) .* (VRmax - VRmin);
                    if r < 0.5
                        new_pos(i, :) = X_rand + E * abs(E * best_pos - X_rand);
                    else
                        new_pos(i, :) = X_rand - E * abs(E * best_pos - X_rand);
                    end
                end
            end

            % Ensure the new position is within bounds
            new_pos(i, :) = max(min(new_pos(i, :), VRmax), VRmin);
        end

        % Evaluate new positions
        new_out = feval('benchmark', new_pos, sfn, 0);
        total_evals = total_evals + ps;

        % Update population positions and fitness
        pos = new_pos;
        out = new_out;

        % Update the global best position found so far
        [current_bestval, best_idx] = min(out);
        if current_bestval < bestval
            bestval = current_bestval;
            best_pos = pos(best_idx, :);
        end

        % Track best fitness after each evaluation within bounds of FE
        for j = total_evals - ps + 1:total_evals
            if j <= FE
                tr(j) = bestval;
            end
        end

        % Early stopping check based on relative improvement
        if useRelStop && total_evals >= evalWindow
            rel_improvement = abs(best_prev_eval - bestval) / max(1, abs(best_prev_eval));
            if rel_improvement < relTol
                tr(total_evals + 1:FE) = bestval;  % Fill remaining entries
                res = [best_pos'; bestval];
                varargout{1} = 1:FE;
                varargout{2} = tr;
                varargout{3} = total_evals;
                return;
            end
            best_prev_eval = bestval;
        end

        % Stop if total evaluations reach FE
        if total_evals >= FE
            break;
        end
    end

    % Fill remaining `tr` entries if FE limit is reached without early stopping
    if total_evals < FE
        tr(total_evals + 1:FE) = bestval;
    end

    % Final output
    res = [best_pos'; bestval];
    varargout{1} = 1:FE;
    varargout{2} = tr;
    varargout{3} = total_evals;
end
