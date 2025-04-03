function [res, varargout] = EHO(I, D, range, FE, useRelStop, relTol, evalWindow)
    % African Elephant Optimization Algorithm (AEO)

    % Initialize parameters
    ps = 30;  % Population size
    sfn = I;  % Function index
    VRmin = repmat(range(1), 1, D);
    VRmax = repmat(range(2), 1, D);

    % Initialize elephant population and evaluate fitness
    pos = rand(ps, D) .* (VRmax - VRmin) + VRmin;
    out = feval('benchmark', pos, sfn, 1);
    
    % Identify the best initial position
    [bestval, best_idx] = min(out);
    best_pos = pos(best_idx, :);

    % Initialize evaluation tracking
    tr = NaN(1, FE);            % Track best fitness after each evaluation
    tr(1:ps) = bestval;
    total_evals = ps;
    best_prev_eval = bestval;   % Track previous best for relative improvement

    % Main loop for function evaluations
    while total_evals < FE
        % Elephant behavior parameters
        alpha = rand();  % Random parameter for exploration
        beta = rand();   % Random parameter for attraction

        for i = 1:ps
            % Exploration or exploitation phase
            if rand() < 0.5
                % Exploration: move towards a random elephant
                rand_elephant = randi(ps);
                pos(i, :) = pos(rand_elephant, :) + beta * (pos(rand_elephant, :) - pos(i, :));
            else
                % Exploitation: move towards the best position
                pos(i, :) = pos(i, :) + alpha * (best_pos - pos(i, :));
            end

            % Ensure the new position is within bounds
            pos(i, :) = max(min(pos(i, :), VRmax), VRmin);
            out(i) = feval('benchmark', pos(i, :), sfn, 0);
            total_evals = total_evals + 1;

            % Update best position if the new solution is better
            if out(i) < bestval
                bestval = out(i);
                best_pos = pos(i, :);
            end

            % Track best fitness after each evaluation if within FE bounds
            if total_evals <= FE
                tr(total_evals) = bestval;
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
