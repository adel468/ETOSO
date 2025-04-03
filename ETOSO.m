% Developed By Prof. Adel Ben Abdennour
% Rev 2.0, April 2025
% ayhem123@yahoo.com

% ETOSO.m - Enhanced Team Oriented Swarm Optimization (ETOSO)
% Description:
% ETOSO is a swarm optimization algorithm implementing both exploration and
% exploitation with a randomized percentage of explorers. Exploiters converge
% on the best-known position, while explorers search based on neighbor information.
% It randomizes a percentage of explorers and utilizes linear exploitation.
% I: Function number identifier for benchmark (Integer)
% D: Number of dimensions (Integer)
% range: [min, max] for search space (Vector of length 2)
% FE: Total Function Evaluations allowed (Integer)
% useRelStop: Boolean flag for relative stopping criteria (Boolean)
% relTol: Relative tolerance for stopping criteria (Double)
% evalWindow: Window size for relative improvement check (Integer)
% ps: Population size (Integer)
% sfn: Function identifier, same as I (Integer)
% n_mut: Number of worst explorers to be randomized (Integer)
% VRmin: Minimum value for each dimension (Vector of length D)
% VRmax: Maximum value for each dimension (Vector of length D)
% pos: Initial positions (Matrix ps x D)
% out: Function evaluation(initial) (Vector of length ps)
% bestval: Best value found so far (Double)
% bstidx: Index of best value (Integer)
% best: Global best position (Vector of length D)
% pbest: Personal best positions (Matrix ps x D)
% pbestval: Personal best fitness (Vector of length ps)
% tr: Store best fitness at each evaluation (Vector of length FE)
% total_evals: Evaluation counter (Integer)
% best_prev_eval: Previous best value (Double)
% pos2: Exploiters' positions (Matrix ps/2 x D)
% ff2: Exploiters' fitness values (Vector of length ps/2)
% out2: Explorers' fitness values (Vector of length ps/2)
% nbestval: Best values for explorers (Vector of length ps/2)
% nbest: Best positions for explorers (Matrix ps/2 x D)
% hhh: Linear weight increment vector (Vector of length ps/2)
% a1: Initial weight factor (Double)
% a2: Increment weight factor (Double)
% WI: Linear increase of weight (Vector of length ps/2)
% wnew: Current weight vector (Vector of length ps/2)
% ii: Dimension loop counter (Integer)
% jj: Exploiter loop counter (Integer)
% xpltidx: Current exploiter index (Integer)
% d: Random dimension (Integer)
% xplridx: Explorer index (Integer)
% pnidx: Previous exploiter index (Integer)
% nnidx: Next exploiter index (Integer)
% pbvtmp: Neighbor fitness values (Vector of length 3)
% pbnidx: Best neighbor index (Integer)
% tmpbest: Best neighbor position (Vector of length D)
% tmpbestval: Best neighbor value (Double)
% w_xplr: Explorer update weight (Double)
% mv_xplt: Exploiter movement (Double)
% mv_xplr: Explorer movement (Vector of length D)
% out1: Exploiters' fitness (Vector of length ps/2)
% sidx: Sorted fitness indices (Vector of length ps/2)
% pidx: Logical index for pbest update (Logical vector of length ps)
% k: Inner loop counter (Integer)
% improved: Logical index, improved pbest (Logical vector of length ps)
% rel_improvement: Relative improvement (Double)
% res: Result matrix (Matrix (D+1) x 1)
% varargout{1}: Evaluation count vector (Vector of length FE)
% varargout{2}: Fitness trace vector (Vector of length FE)
% varargout{3}: Total evaluations (Integer)

function [res, varargout] = ETOSO(I, D, range, FE, useRelStop, relTol, evalWindow)
% Input Parameters:
%   I      - Function number identifier for benchmark
%   D      - Number of dimensions
%   range  - [min, max] for search space
%   FE     - Total Function Evaluations allowed

%%%%%%%%%%%%%%%%% PSO PARAMS %%%%%%%%%%%%%%%%%%%%%
ps = 30;  % Population size
sfn = I;  % Function identifier
n_mut = round((ps/2) *0);  % Number of worst explorers  to be randomized
VRmin = repmat(range(1), 1, D);  % Minimum value for each dimension
VRmax = repmat(range(2), 1, D);  % Maximum value for each dimension

%%%%%%%%%%%%%%% Initialization %%%%%%%%%%%%%%%%%%%%%%%
% Initialize minimum and maximum values based on the function index I

% Randomize initial population across the whole search space
pos = VRmin + rand(ps, D) .* (VRmax - VRmin);  % Initial positions
[out, ~, ~, ~] = benchmark(pos, sfn, 1);  % Function evaluation(initial)
[bestval, bstidx] = min(out);  % Evaluate best value & position
best = pos(bstidx, :);  % Global best position
out = benchmark(pos, sfn, 1); % Initial evaluation
[bestval, bstidx] = min(out);
best = pos(bstidx, :);  % Global best position
pbest = pos;             % Personal best positions
pbestval = out;          % Personal best fitness
tr = NaN(1, FE);         % Store best fitness at each evaluation
tr(1:ps) = bestval;
total_evals = ps;        % Start evaluation counter
best_prev_eval = bestval; % Store previous best value for relative improvement check

%---------------- Explotation ----------------------------
pos2 = repmat(best, ps/2, 1);  % Set positions of exploiters to Gbest
ff2 = repmat(bestval, ps/2, 1);  % Set fitness values of exploiters
%---------------- Exploration ----------------------------
out2 = out(ps/2 + 1:ps);  % Extract out values for explorers
pbest = [pos2; pos((ps/2 + 1:ps), :)];  % Combine best positions
pbestval = [ff2; out2];  % Combine best values
nbestval = out2;  % Best values for explorers
nbest = pos((ps/2 + 1:ps), :);  % Best positions for explorers
%   tbst = repmat(bestval, 1, ps);
hhh = 1:ps/2;

% Linear weight increment factors for exploitation
% a1 = 0.035;  % Initial weight factor
% a2 = 0.001;  % Increment weight factor
a1=0.996/(ps-2);
a2=0.001;
WI = a1 * (hhh - 1) + a2;  % Linear increase of weight for exploitation
wnew = WI;

%%%%%%%%%%%%%%%%% Start of Iteration %%%%%%%%%%%
while total_evals < FE
    % Define parameters for stagnation check
    for ii = 1:D  % Repeat for each dimension
        if total_evals >= FE
            break;
        end

        pos(1:ps/2, :) = repmat(best, ps/2, 1);  % Set exploiters' positions to Gbest location

        for jj = 1:ps/2  % Iterate through each exploiter
            xpltidx = jj;  % Current exploiter index
            d = randi([1 D]);  % Select a random dimension
            xplridx = jj + ps/2;  % Index for the explorer

            % Determine neighboring positions based on indexing
            if jj == 1
                pnidx = ps;  % Wrap around to the end of the array
            else
                pnidx = xplridx - 1;  % Previous exploiter index
            end

            nnidx = (rem(jj, ps/2) + 1) + ps/2;  % Next exploiter index
            pbvtmp = [pbestval(pnidx); pbestval(xplridx); pbestval(nnidx)];  % Collect values for comparison
            [~, pbnidx] = min(pbvtmp);  % Find the index of the best value among neighbors

            % Update nbest based on the best neighbor found
            if pbnidx == 1
                tmpbest = pbest(pnidx, :);
                tmpbestval = pbestval(pnidx);
            elseif pbnidx == 2
                tmpbest = pbest(xplridx, :);
                tmpbestval = pbestval(xplridx);
            elseif pbnidx == 3
                tmpbest = pbest(nnidx, :);
                tmpbestval = pbestval(nnidx);
            end

            nbest(xplridx, :) = tmpbest;  % Update position of explorer
            nbestval(xplridx) = tmpbestval;  % Update value of explorer

            %------------------------ Position Update -------------------------
            w_xplr = abs(bestval / (1 + max(out2)));  % Weight for the explorer update
            mv_xplt = wnew(xpltidx) .* randn * (VRmax(d) - VRmin(d));  % Movement for exploiter
            mv_xplr = w_xplr .* rand(1, D) .* ((nbest(xplridx, :) - pos(xplridx, :)));  % Movement for explorer

            % Update positions
            pos(xpltidx, d) = pos(xpltidx, d) + mv_xplt;  % Update exploiter position
            pos(xplridx, :) = mv_xplr;  % Update explorer position

            % Randomizing the last 50% of explorers
            if xplridx > (ps - n_mut)
                pos(xplridx, :) = VRmin + (rand(1, D) .* (VRmax - VRmin));  % Random position assignment
                display('************MUTATION*********')
            end

            % Enforce boundary constraints
            pos(jj, :) = max(min(pos(jj, :), VRmax), VRmin);
            pos(xplridx, :) = max(min(pos(xplridx, :), VRmax), VRmin);
        end

        % Evaluate the function based on new positions
        out = benchmark(pos, sfn, 0);  % Function evaluation
        out1 = out(1:ps/2);
        out2 = out(ps/2 + 1:ps);  % Split evaluations
        % Update weights based on sort order of exploitation
        [~, sidx] = sort(out1);  % Sort the indices of the exploiter fitness values

        % Update the weight values based on sorted indices
        wnew = WI(sidx);  % Assign new weights to exploiters based on sorted order

        %------------------- Update Personal Bests --------------------------
        pidx = pbestval >= out;  % Create a logical index array
        pbest(pidx, :) = pos(pidx, :);  % Update the personal best positions
        pbestval(pidx) = out(pidx);  % Update the personal best values
    for k = 1:ps
                total_evals = total_evals + 1;
                if total_evals > FE
                    break;
                end
        % Update best fitness if a new minimum is found
        if out(k) < bestval
            bestval = out(k);
            best = pos(k, :);
        end
        tr(total_evals) = bestval;  % Record the best fitness after each evaluation
    end

    % Update personal bests
    improved = out < pbestval;
    pbest(improved, :) = pos(improved, :);
    pbestval(improved) = out(improved);

    % Check the relative improvement for early stopping
    if useRelStop && total_evals >= evalWindow
        if best_prev_eval == 0 && bestval == 0
            rel_improvement = 0;  % Assume no improvement if both are zero
        elseif best_prev_eval ~= 0
            rel_improvement = abs(best_prev_eval - bestval) / abs(best_prev_eval);
        else
            rel_improvement = Inf;  % Large value if best_prev_eval is zero but bestval is not
        end


        if rel_improvement < relTol
            %   fprintf('Relative improvement threshold reached. Stopping at evaluation %d with best fitness %.5f\n', total_evals, bestval);

            % Early stopping - fill remaining `tr` with bestval
            tr(total_evals + 1:FE) = bestval;
            res = [best'; bestval];
            varargout{1} = 1:FE;
            varargout{2} = tr;
            varargout{3} = total_evals;
            return;
        end
        best_prev_eval = bestval;  % Update previous best for next comparison
    end

    % Break if `FE` evaluations have been reached
    if total_evals >= FE
        break;
    end
    end
end


% Fill remaining `tr` values if FE limit is reached without early stopping
 if total_evals < FE
     tr(total_evals + 1:FE) = bestval;
 end

% Final output values
res = [best'; bestval];
varargout{1} = 1:FE;
varargout{2} = tr;
varargout{3} = total_evals;
