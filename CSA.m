function [res, varargout] = CSA(I, D, range, FE, useRelStop, relTol, evalWindow)
    % Crow Search Algorithm (CSA)
    
    % Initialize parameters
    ps = 30;  % Population size
    sfn = I;  % Function index
    VRmin = repmat(range(1), 1, D);
    VRmax = repmat(range(2), 1, D);
    
    AP = 0.1;  % Awareness probability
    FL = 2;    % Flight length

    % Initialize crow population and memory
    pos = rand(ps, D) .* (VRmax - VRmin) + VRmin;
    memory = pos;  % Memory to store the best known positions
    out = benchmark(pos, sfn, 1);
    
    % Identify the best initial position
    [bestval, best_idx] = min(out);
    best_pos = pos(best_idx, :);

    % Initialize vectors to store the best fitness values at each function evaluation
    tr = NaN(1, FE);            % Record best fitness after each evaluation
    tr(1:ps) = bestval;          % Store initial fitness values
    total_evals = ps;            % Start evaluation counter
    best_prev_eval = bestval;    % Track previous best for relative improvement

    % Main loop for function evaluations
    while total_evals < FE
        for i = 1:ps
            r = rand();  % Random number to decide the movement type
            if r >= AP
                % Follow a random crow
                rand_j = randi([1, ps]);
                new_pos = pos(i, :) + FL * rand(1, D) .* (memory(rand_j, :) - pos(i, :));
            else
                % Move randomly
                new_pos = VRmin + rand(1, D) .* (VRmax - VRmin);
            end
            
            % Ensure the new position is within bounds
            new_pos = max(min(new_pos, VRmax), VRmin);
            new_out = benchmark(new_pos, sfn, 0);
            total_evals = total_evals + 1;

            % Update memory if new position is better
            if new_out < out(i)
                memory(i, :) = new_pos;
                out(i) = new_out;
            end

            % Update global best if this solution is the best found so far
            if new_out < bestval
                bestval = new_out;
                best_pos = new_pos;
            end

            % Record the best fitness for each evaluation
            if total_evals <= FE
                tr(total_evals) = bestval;
            end

            % Early stopping check
            if useRelStop && total_evals >= evalWindow
                rel_improvement = abs(best_prev_eval - bestval) / max(1, abs(best_prev_eval));
                if rel_improvement < relTol
                    tr(total_evals + 1:FE) = bestval;
                    res = [best_pos'; bestval];
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
    end

    % Fill remaining `tr` values if FE limit is reached without early stopping
    if total_evals < FE
        tr(total_evals + 1:FE) = bestval;
    end

    % Final output
    res = [best_pos'; bestval];
    varargout{1} = 1:FE;
    varargout{2} = tr;
    varargout{3} = total_evals;
end
