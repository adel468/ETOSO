function [res, varargout] = GOA(I, D, range, FE, useRelStop, relTol, evalWindow)
    % Grasshopper Optimization Algorithm (GOA)

    % Initialize parameters
    ps = 30;               % Population size
    sfn = I;               % Function index
    VRmin = range(1);
    VRmax = range(2);
    c_max = 1;             % Maximum value of the c parameter
    c_min = 0.00001;       % Minimum value of the c parameter

    % Initialize grasshopper population and evaluate initial fitness
    pos = VRmin + rand(ps, D) * (VRmax - VRmin);
    out = feval('benchmark', pos, sfn, 1);

    % Initialize best fitness tracking
    [bestval, best_idx] = min(out);
    best_pos = pos(best_idx, :);

    % Initialize record-keeping for function evaluations and best values
    tr = NaN(1, FE);                % Track best fitness after each evaluation
    tr(1:ps) = bestval;
    total_evals = ps;
    best_prev_eval = bestval;       % Track previous best for relative improvement

    % Main loop for function evaluations
    while total_evals < FE
        % Update the `c` parameter linearly
        c = c_max - total_evals * ((c_max - c_min) / FE);

        % Update positions of each grasshopper
        new_pos = pos;
        for i = 1:ps
            % Grasshopper interaction forces
            grasshopper_force = zeros(1, D);
            for j = 1:ps
                if i ~= j
                    distance = norm(pos(j, :) - pos(i, :));
                    s_ij = 2 * exp(-distance) * (0.5 * (sin(distance) / distance) - cos(distance));
                    direction = (pos(j, :) - pos(i, :)) / max(distance, eps); % Avoid division by zero
                    grasshopper_force = grasshopper_force + c * (s_ij .* direction);
                end
            end
            new_pos(i, :) = grasshopper_force + best_pos;

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

        % Track best fitness after each evaluation if within bounds of FE
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
