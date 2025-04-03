function [res, varargout] = FPA(I, D, range, FE, useRelStop, relTol, evalWindow)
    % Flower Pollination Algorithm (FPA)
    
    % Initialize parameters
    ps = 30;              % Population size
    sfn = I;              % Function index
    VRmin = range(1);
    VRmax = range(2);
    p = 0.8;              % Switch probability between global and local pollination

    % Initialize flower population and evaluate initial fitness
    pos = VRmin + rand(ps, D) * (VRmax - VRmin);
    out = feval('benchmark', pos, sfn, 1);

    % Initialize best fitness tracking
    [bestval, bstidx] = min(out);
    best = pos(bstidx, :);

    % Initialize record-keeping for function evaluations and best values
    tr = NaN(1, FE);                % Track best fitness after each evaluation
    tr(1:ps) = bestval;
    total_evals = ps;
    best_prev_eval = bestval;       % Track previous best for relative improvement

    % Main loop for function evaluations
    while total_evals < FE
        for i = 1:ps
            % Pollination process (Global or Local)
            if rand < p
                % Global pollination using Levy flight
                L = levy_flight(D);
                new_pos = pos(i, :) + L .* (best - pos(i, :));
            else
                % Local pollination
                j = randi(ps);
                k = randi(ps);
                while j == i, j = randi(ps); end
                while k == i || k == j, k = randi(ps); end
                new_pos = pos(i, :) + rand * (pos(j, :) - pos(k, :));
            end

            % Ensure the new position is within bounds
            new_pos = max(min(new_pos, VRmax), VRmin);
            new_out = feval('benchmark', new_pos, sfn, 0);
            total_evals = total_evals + 1;

            % Update position if there’s improvement
            if new_out < out(i)
                pos(i, :) = new_pos;
                out(i) = new_out;
                if new_out < bestval
                    bestval = new_out;
                    best = new_pos;
                end
            end

            % Track best fitness after each evaluation if within bounds of FE
            if total_evals <= FE
                tr(total_evals) = bestval;
            end

            % Early stopping check based on relative improvement
            if useRelStop && total_evals >= evalWindow
                rel_improvement = abs(best_prev_eval - bestval) / max(1, abs(best_prev_eval));
                if rel_improvement < relTol
                    tr(total_evals + 1:FE) = bestval;  % Fill remaining entries
                    res = [best'; bestval];
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
    res = [best'; bestval];
    varargout{1} = 1:FE;
    varargout{2} = tr;
    varargout{3} = total_evals;
end

function L = levy_flight(D)
    % Levy flight calculation
    beta = 1.5;
    sigma = (gamma(1 + beta) * sin(pi * beta / 2) / ...
            (gamma((1 + beta) / 2) * beta * 2^((beta - 1) / 2)))^(1 / beta);
    u = randn(1, D) * sigma;
    v = randn(1, D);
    L = u ./ abs(v).^(1 / beta);
end
