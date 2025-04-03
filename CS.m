function [res, varargout] = CS(I, D, range, FE, useRelStop, relTol, evalWindow)
    % Cuckoo Search Algorithm (CS3)
    fncton = I;               % Function index
    ps = 30;                  % Population size
    pa = 0.25;                % Probability of alien egg discovery
    beta = 1;                 % Levy flight exponent
    VRmin = repmat(range(1), 1, D);
    VRmax = repmat(range(2), 1, D);

    % Initialize nests
    pos = rand(ps, D) .* (VRmax - VRmin) + VRmin;
    out = feval('benchmark', pos, fncton, 1);

    % Initialize best fitness tracking
    [bestval, best_idx] = min(out);
    best = pos(best_idx, :);

    % Initialize record-keeping for function evaluations and best values
    tr = NaN(1, FE);               % Store best fitness after each evaluation
    tr(1:ps) = bestval;             % Initial fitness values
    total_evals = ps;               % Start evaluation counter
    best_prev_eval = bestval;       % Track previous best for relative improvement

    % Main loop for function evaluations
    while total_evals < FE
        % Generate new solutions by Levy flights
        new_pos = pos;
        for i = 1:ps
            step_size = levy(D, beta);
            new_pos(i, :) = pos(i, :) + step_size .* (pos(i, :) - pos(randi(ps), :));
            % Ensure the positions do not exceed the search bounds
            new_pos(i, :) = max(min(new_pos(i, :), VRmax), VRmin);
        end

        % Evaluate new positions
        new_out = feval('benchmark', new_pos, fncton, 0);
        total_evals = total_evals + ps;

        % Select better solutions
        improved = new_out < out;
        pos(improved, :) = new_pos(improved, :);
        out(improved) = new_out(improved);

        % Update global best if found among newly generated solutions
        [min_out, bstidx] = min(out);
        if min_out < bestval
            bestval = min_out;
            best = pos(bstidx, :);
        end

        % Record the best fitness for each evaluation within bounds of FE
        for j = total_evals - ps + 1:total_evals
            if j <= FE
                tr(j) = bestval;
            end
        end

        % Abandon a fraction of worst nests and replace with new random solutions
        for i = 1:ps
            if rand < pa && total_evals < FE
                pos(i, :) = rand(1, D) .* (VRmax - VRmin) + VRmin;
                out(i) = feval('benchmark', pos(i, :), fncton, 0);
                total_evals = total_evals + 1;

                % Update global best if a better solution is found
                if out(i) < bestval
                    bestval = out(i);
                    best = pos(i, :);
                end

                % Store best fitness if within FE
                if total_evals <= FE
                    tr(total_evals) = bestval;
                end
            end
        end

        % Early stopping check
        if useRelStop && total_evals >= evalWindow
            rel_improvement = abs(best_prev_eval - bestval) / max(1, abs(best_prev_eval));
            if rel_improvement < relTol
                tr(total_evals + 1:FE) = bestval;
                res = [best'; bestval];
                varargout{1} = 1:FE;
                varargout{2} = tr;
                varargout{3} = total_evals;
                return;
            end
            best_prev_eval = bestval;
        end

        % Check if the maximum number of function evaluations has been reached
        if total_evals >= FE
            break;
        end
    end

    % Fill remaining `tr` values if FE limit is reached without early stopping
    if total_evals < FE
        tr(total_evals + 1:FE) = bestval;
    end

    % Final output
    res = [best'; bestval];
    varargout{1} = 1:FE;
    varargout{2} = tr;
    varargout{3} = total_evals;
end

function step = levy(D, beta)
    % Levy flight generation
    sigma = (gamma(1 + beta) * sin(pi * beta / 2) / ...
            (gamma((1 + beta) / 2) * beta * 2^((beta - 1) / 2)))^(1 / beta);
    u = randn(1, D) * sigma;
    v = randn(1, D);
    step = u ./ abs(v).^(1 / beta);
end
