function [res, varargout] = RRO(I, D, range, FE, useRelStop, relTol, evalWindow)
    % Raven Roosting Optimization (RRO)
    
    % Initialize parameters
    pop_size = 30;                      % Population size
    functionIndex = I;                  % Benchmark function index
    R = (range(2) - range(1)) / 2;      % Search radius
    Rpcpt = 3.6 * R * sqrt(D);          % Radius of perception
    Rleader = 1.8 * R * sqrt(D);        % Radius of leader's vicinity
    Nsteps = 5;                         % Number of steps for each raven
    Perc_follow = 0.2;                  % Proportion of followers
    Prob_stop = 0.1;                    % Probability of stopping
    
    % Initialize raven population and evaluate initial fitness
    VRmin = range(1);
    VRmax = range(2);
    pos = VRmin + rand(pop_size, D) * (VRmax - VRmin);
    fitness = feval('benchmark', pos, functionIndex, 1);

    % Initialize best fitness tracking
    [best_fitness, best_idx] = min(fitness);
    best_pos = pos(best_idx, :);

    % Initialize record-keeping for function evaluations and best values
    tr = NaN(1, FE);                    % Track best fitness after each evaluation
    tr(1:pop_size) = best_fitness;
    total_evals = pop_size;
    best_prev_eval = best_fitness;      % Track previous best for relative improvement

    % Main loop for function evaluations
    while total_evals < FE
        % Identify the LEADER position
        LEADER = best_pos;  % Currently best position found
        n_followers = round(Perc_follow * pop_size);
        followers_indices = randperm(pop_size, n_followers);

        % Update followers
        for i = followers_indices
            for step = 1:Nsteps
                r = rand();
                if r < Prob_stop
                    % Stopping behavior within perception radius
                    pos(i, :) = pos(i, :) + randn(1, D) .* (Rpcpt * 0.1);
                else
                    % Move towards LEADER's vicinity
                    direction = randn(1, D);
                    pos(i, :) = LEADER + Rleader * (direction / norm(direction));
                end

                % Ensure new position is within bounds
                pos(i, :) = max(min(pos(i, :), VRmax), VRmin);
                fitness(i) = feval('benchmark', pos(i, :), functionIndex, 0);
                total_evals = total_evals + 1;

                % Update global best if new position improves
                if fitness(i) < best_fitness
                    best_fitness = fitness(i);
                    best_pos = pos(i, :);
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
        if total_evals >= FE
            break;
        end

        % Update non-followers with exploration/exploitation behavior
        non_followers_indices = setdiff(1:pop_size, followers_indices);
        for i = non_followers_indices
            C = 1 - (total_evals / FE);  % Decreasing attraction factor

            if rand() < 0.5
                % Exploration behavior
                random_raven = randi(pop_size);
                pos(i, :) = pos(random_raven, :) + C * (rand() * Rpcpt + VRmin);
            else
                % Exploitation behavior towards best position
                pos(i, :) = pos(i, :) + C * (best_pos - pos(i, :)) + (randn(1, D) .* (VRmax - VRmin) * 0.1);
            end

            % Ensure new position is within bounds
            pos(i, :) = max(min(pos(i, :), VRmax), VRmin);
            fitness(i) = feval('benchmark', pos(i, :), functionIndex, 0);
            total_evals = total_evals + 1;

            % Update global best if new position improves
            if fitness(i) < best_fitness
                best_fitness = fitness(i);
                best_pos = pos(i, :);
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
