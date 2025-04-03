function [res, varargout] = MFO(I, D, range, FE, useRelStop, relTol, evalWindow)
    % Moth-Flame Optimization (MFO) Algorithm
    
    % Initialize parameters
    populationSize = 30;                % Number of moths (population size)
    spiralConstant = -1;                % Constant for defining the spiral shape
    functionIndex = I;

    VRmin = range(1);
    VRmax = range(2);

    % Initialize moth population positions and calculate initial fitness
    positions = VRmin + rand(populationSize, D) * (VRmax - VRmin);
    fitnessValues = feval('benchmark', positions, functionIndex, 1);

    % Determine initial flames (best moth positions)
    [bestFitness, sortedIndices] = min(fitnessValues);
    bestPosition = positions(sortedIndices, :);

    % Initialize record-keeping for function evaluations and best values
    tr = NaN(1, FE);                    % Track best fitness after each evaluation
    tr(1:populationSize) = bestFitness;
    total_evals = populationSize;
    best_prev_eval = bestFitness;       % Track previous best for relative improvement

    % Main loop for function evaluations
    while total_evals < FE
        flameCount = round(populationSize - (total_evals / FE) * (populationSize - 1));

        % Update positions of each moth
        new_positions = positions;
        for moth = 1:populationSize
            for dim = 1:D
                % Calculate new position using a logarithmic spiral
                distanceToFlame = abs(bestPosition(dim) - positions(moth, dim));
                spiralFactor = (2 * rand - 1);
                newPosition = distanceToFlame * exp(spiralConstant * spiralFactor) * cos(spiralFactor * 2 * pi) + bestPosition(dim);

                % Ensure the new position is within bounds
                new_positions(moth, dim) = max(min(newPosition, VRmax), VRmin);
            end
            % Evaluate the new position
            new_fitness = feval('benchmark', new_positions(moth, :), functionIndex, 0);
            total_evals = total_evals + 1;

            % Update position and fitness if the new fitness is better
            if new_fitness < fitnessValues(moth)
                positions(moth, :) = new_positions(moth, :);
                fitnessValues(moth) = new_fitness;
                % Update the global best fitness if this position is the new best
                if new_fitness < bestFitness
                    bestFitness = new_fitness;
                    bestPosition = new_positions(moth, :);
                end
            end

            % Store the best fitness after each evaluation if within bounds of FE
            if total_evals <= FE
                tr(total_evals) = bestFitness;
            end

            % Early stopping check based on relative improvement
            if useRelStop && total_evals >= evalWindow
                rel_improvement = abs(best_prev_eval - bestFitness) / max(1, abs(best_prev_eval));
                if rel_improvement < relTol
                    tr(total_evals + 1:FE) = bestFitness;  % Fill remaining entries
                    res = [bestPosition'; bestFitness];
                    varargout{1} = 1:FE;
                    varargout{2} = tr;
                    varargout{3} = total_evals;
                    return;
                end
                best_prev_eval = bestFitness;
            end

            % Stop if total evaluations reach FE
            if total_evals >= FE
                break;
            end
        end

        % Sort moths based on fitness to determine flames for the next epoch
        [sorted_fitness, sortedIndices] = sort(fitnessValues);
        flames = positions(sortedIndices(1:flameCount), :);
    end

    % Fill remaining `tr` entries if FE limit is reached without early stopping
    if total_evals < FE
        tr(total_evals + 1:FE) = bestFitness;
    end

    % Final output
    res = [bestPosition'; bestFitness];
    varargout{1} = 1:FE;
    varargout{2} = tr;
    varargout{3} = total_evals;
end
