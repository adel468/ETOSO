function [res, varargout] = BEE(I, D, range, FE, useRelStop, relTol, evalWindow)
    % Parameters for the algorithm
    n = 30;   % Number of scout bees (initial solutions)
    m = min(5, n);    % Number of selected sites, cannot exceed n
    e = 2;    % Number of elite sites
    nep = 5;  % Number of elite site bees
    nsp = 2;  % Number of selected site bees
    ngh = .1;  % Initial patch size

    % Initialize population
    pos = rand(n, D) .* (range(2) - range(1)) + range(1);
    fitness = benchmark(pos, I, 1);
    [best_fitness, best_idx] = min(fitness);
    best = pos(best_idx, :);

    % Initialize vectors to store best fitness values at each evaluation
    tr = NaN(1, FE);               % Record best fitness at each function evaluation
    tr(1:n) = best_fitness;         % Store initial best fitness
    total_evals = n;                % Start evaluation counter
    best_prev_eval = best_fitness;  % Track the previous best fitness for relative improvement

    % Main loop for function evaluations
    while total_evals < FE
        % Update m based on the population size
        current_m = min(m, size(pos, 1));

        % Select m sites based on fitness
        [selected_sites, selected_fitness, indices] = selectSites(pos, fitness, current_m);

        % Explore elite sites
        for i = 1:min(e, current_m)
            for j = 1:nep
                candidate = selected_sites(i, :) + ngh * randn(1, D);
                candidate = max(candidate, range(1)); % Ensure within bounds
                candidate = min(candidate, range(2));
                candidate_fitness = benchmark(candidate, I, 1);
                total_evals = total_evals + 1;

                % Update selected site with improved candidate
                if candidate_fitness < selected_fitness(i)
                    selected_sites(i, :) = candidate;
                    selected_fitness(i) = candidate_fitness;
                end

                % Update global best fitness and position
                if candidate_fitness < best_fitness
                    best_fitness = candidate_fitness;
                    best = candidate;
                end

                % Record best fitness for each evaluation
                if total_evals <= FE
                    tr(total_evals) = best_fitness;
                end

                % Early stopping check
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

                if total_evals >= FE
                    break;
                end
            end
            if total_evals >= FE
                break;
            end
        end

        % Explore non-elite sites
        for i = (e + 1):current_m
            for j = 1:nsp
                candidate = selected_sites(i, :) + ngh * randn(1, D);
                candidate = max(candidate, range(1)); % Ensure within bounds
                candidate = min(candidate, range(2));
                candidate_fitness = benchmark(candidate, I, 1);
                total_evals = total_evals + 1;

                % Update selected site with improved candidate
                if candidate_fitness < selected_fitness(i)
                    selected_sites(i, :) = candidate;
                    selected_fitness(i) = candidate_fitness;
                end

                % Update global best fitness and position
                if candidate_fitness < best_fitness
                    best_fitness = candidate_fitness;
                    best = candidate;
                end

                % Record best fitness for each evaluation
                if total_evals <= FE
                    tr(total_evals) = best_fitness;
                end

                % Early stopping check
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

                if total_evals >= FE
                    break;
                end
            end
            if total_evals >= FE
                break;
            end
        end

        % Update the population with the selected evaluated sites
        pos = selected_sites;
        fitness = selected_fitness;
    end

    % Fill remaining `tr` values if FE limit is reached
    if total_evals < FE
        tr(total_evals + 1:FE) = best_fitness;
    end

    % Return results
    res = [best'; best_fitness];
    varargout{1} = 1:FE;
    varargout{2} = tr;
    varargout{3} = total_evals;
end

function [selected_sites, selected_fitness, indices] = selectSites(pos, fitness, m)
    % Select sites based on fitness
    [sorted_fitness, indices] = sort(fitness);
    selected_sites = pos(indices(1:m), :);
    selected_fitness = sorted_fitness(1:m);
end
