function [res, varargout] = DE(I, D, range, FE, useRelStop, relTol, evalWindow)
    % Differential Evolution (DE)
    
    % Initialize parameters
    pop_size = 30;  % Population size
    CR = 0.9;       % Crossover probability
    F = 0.8;        % Mutation factor
    fncton = I;     % Function index
    VRmin = range(1);
    VRmax = range(2);
    
    % Initialize population and evaluate fitness
    pos = VRmin + rand(pop_size, D) * (VRmax - VRmin);
    fitness = benchmark(pos, fncton, 1);
    [bestval, best_idx] = min(fitness);
    best = pos(best_idx, :);

    % Initialize evaluation tracking
    tr = NaN(1, FE);             % Track best fitness after each evaluation
    tr(1:pop_size) = bestval;
    total_evals = pop_size;
    best_prev_eval = bestval;    % Track previous best for relative improvement

    % Main DE loop
    while total_evals < FE
        for i = 1:pop_size
            % Mutation and crossover steps
            idxs = randperm(pop_size, 3);
            v = pos(idxs(1), :) + F * (pos(idxs(2), :) - pos(idxs(3), :));
            v = max(min(v, VRmax), VRmin); % Boundary control

            % Crossover
            cross_points = rand(1, D) < CR;
            if ~any(cross_points)
                cross_points(randi(D)) = true;
            end
            trial = pos(i, :);
            trial(cross_points) = v(cross_points);

            % Evaluate trial vector
            trial_fitness = benchmark(trial, fncton, 0);
            total_evals = total_evals + 1;

            % Selection: replace if trial solution is better
            if trial_fitness < fitness(i)
                pos(i, :) = trial;
                fitness(i) = trial_fitness;
                if trial_fitness < bestval
                    bestval = trial_fitness;
                    best = trial;
                end
            end

            % Track best fitness for each evaluation within FE bounds
            if total_evals <= FE
                tr(total_evals) = bestval;
            end

            % Early stopping check based on relative improvement
            if useRelStop && total_evals >= evalWindow
                rel_improvement = abs(best_prev_eval - bestval) / max(1, abs(best_prev_eval));
                if rel_improvement < relTol
                    tr(total_evals + 1:FE) = bestval;  % Fill remaining values
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

    % Fill any remaining entries in `tr` if early stopping was not triggered
    if total_evals < FE
        tr(total_evals + 1:FE) = bestval;
    end

    % Final output
    res = [best'; bestval];
    varargout{1} = 1:FE;
    varargout{2} = tr;
    varargout{3} = total_evals;
end
