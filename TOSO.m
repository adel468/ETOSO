% toso.m, Team Oriented Swarm Optimization,
% Rev 2.0, March 2012
% Developed By Prof. Adel Ben Abdennour & Faizal Hafiz
% Team Oriented Swarm Optimization (TOSO) Algorithm
% ayhem123@yahoo.com

% Objective:
% The TOSO algorithm is a population-based optimization technique designed to
% solve complex optimization problems by leveraging the principles of swarm
% intelligence. It is particularly effective for high-dimensional search
% spaces and is inspired by the behavior of social organisms such as birds or
% fish.
%
% Key Features:
% 1. Swarm-Based Optimization:
%    - TOSO uses a population of particles (swarm) to explore the search space.
%    - Each particle represents a potential solution to the optimization problem.
%
% 2. Dual Strategy (Exploration & Exploitation):
%    - Exploitation: A subset of particles (exploiters) focuses on refining the
%      best-known solution by staying close to the global best position.
%    - Exploration: Another subset (explorers) searches the broader space to
%      discover potentially better solutions.
%
% 3. Dynamic Weighting:
%    - Particles use dynamically adjusted weights to balance exploration and
%      exploitation.
%    - Weights are updated based on the performance of particles during iterations.
%
% 4. Boundary Handling:
%    - Positions of particles are clipped to ensure they remain within the
%      defined search space bounds.
%
% 5. Early Stopping Mechanism:
%    - The algorithm can terminate early if the relative improvement in the
%      best fitness value falls below a specified tolerance threshold.
%
% Input Parameters:
% - I: Function index for the benchmark function to optimize.
% - D: Dimensionality of the search space.
% - range: A 2-element vector defining the lower and upper bounds of the search space.
% - FE: Maximum number of function evaluations.
% - useRelStop: Boolean flag to enable/disable early stopping based on relative improvement.
% - relTol: Tolerance threshold for relative improvement (used if useRelStop is true).
% - evalWindow: Number of evaluations to consider for checking relative improvement.
%
% Output:
% - res: A vector containing the best solution found and its corresponding fitness value.
% - varargout: Additional outputs including evaluation history, best fitness values,
%   and total number of evaluations performed.
%
% Notes:
% - The algorithm is designed to work with benchmark functions indexed from 1 to 15.
% - Invalid function indices will result in an error.



function [res, varargout]=TOSO(I, D, range, FE, useRelStop, relTol, evalWindow)
%%%%%%%%%%%%%%%% PSO PARAMS %%%%%%%%%%%%%%%%%%%%%
ps=30; sfn=I; pm=0.2;
VRmin = repmat(range(1),1,D); VRmax = repmat(range(2),1,D);
xi=D;  %stpcnt = 0;
%%%%%%%%%%%%%%% INtialization %%%%%%%%%%%%%%%%%%%%%%%
% Initialize minimum and maximum values based on the function index I
if I >= 1 && I <= 15
    tmpmin = repmat(range(1), 1, D);  % Minimum value based on the first element of range
    tmpmax = repmat(range(2), 1, D);  % Maximum value based on the second element of range
else
    error('Invalid function index I.');  % Error handling for invalid indices
end
  
pos = (rand(ps, D) .* (tmpmax - tmpmin)) + tmpmin;  % Randomize Initial Population Over Whole Search Space
[out] = benchmark(pos,sfn,1);                       %%%%% Function Evaluation
[bestval, bstidx] = min(out); best = pos(bstidx,:);                         %%%%% Evluate Best Value & Best Position (Gbest)
tr = NaN(1, FE);         % Store best fitness at each evaluation
tr(1:ps) = bestval;
total_evals = ps;        % Start evaluation counter
best_prev_eval = bestval; % Store previous best value for relative improvement check
%---------------- Explotation----------------------------  
%chgreg = pos(1:ps/2,:); 
pos2 = repmat(best,ps/2,1);                        %%%%% Set Old & New Position of Explioters
ff2 = repmat(bestval,ps/2,1); %ff1 = out(1:ps/2,1);                         %%%%% Set Old & New Fitness of Explioters
%xpltbstval = bestval; xpltbest = best;                                     %%%%% Set Best Value & Best of Exploiters to Gbest
%---------------- Exploration---------------------------- 
out2 = out(ps/2+1:ps);                                                      
%[~, idx11] = min(out2); %xplrbest = pos(idx11,:);                   %%%%% Evluate Best Value & Best Position of Explorers
pbest=[pos2;pos((ps/2+1:ps),:)]; pbestval=[ff2;out2]; 
nbestval=out2; nbest=pos((ps/2+1:ps),:);
tr(1)=bestval;  tbst=repmat(bestval,1,ps); 
% Precompute weight factors to avoid repeating calculations
a1 = 0.001; 
a2 = 0.499; 
a3 = 2;
WI = a1 + (a2 * (exp(a3 * (0:(ps/2 - 1)) / (ps/2 - 1)) - 1) / (exp(a3) - 1));  % Vectorized creation of WI
wnew = WI;
%%%%%%%%%%%%%%%% Start of Iteration %%%%%%%%%%%

  
while total_evals < FE
    for ii=1:xi                                                            %%%%% "D" Times Repeatation
      if total_evals >= FE
            break;
       end

        pos(1:ps/2,:) = repmat(best,ps/2,1);                               %%%%% Set Explioters to Gbest location
        for jj=1:ps/2                                                      %%%%% For whole population
            xpltidx = jj; %randi([1 ps/2]); 
            d = randi([1 D]);                   %%%%% Select Random Dimension of Randomly Selected Xplter Particle
            xplridx = jj+ps/2;
            if jj==1; pnidx=ps; 
            else 
                pnidx = xplridx-1; 
            end
            nnidx=(rem(jj,ps/2)+1)+ps/2; 
% Optimize neighbor comparisons using logical indexing
pbvtmp = [pbestval(pnidx), pbestval(xplridx), pbestval(nnidx)];  % Collect values for comparison
[tmpbestval, pbnidx] = min(pbvtmp);  % Find the index of the best value among neighbors

% Update nbest based on the best neighbor found
switch pbnidx
    case 1
        tmpbest = pbest(pnidx, :); 
    case 2
        tmpbest = pbest(xplridx, :); 
    case 3
        tmpbest = pbest(nnidx, :); 
end
nbest(xplridx,:)=tmpbest;
            nbestval(xplridx)=tmpbestval;
          %------------------------Position Update-------------------------  
            w_xplr=abs(bestval/(1+max(out2)));%1.0; %abs(fxr);
            mv_xplt =wnew(xpltidx).*randn*(VRmax(d)-VRmin(d)); 
            mv_xplr=w_xplr.*rand(1,D).*((nbest(xplridx,:)-pos(xplridx,:))); 
            pos(xpltidx,d) = pos(xpltidx,d)+mv_xplt;%%%  GOOD
            pos(xplridx,:) = mv_xplr;   %GOOD RESULT 
            if rand<=pm
                 pos(xplridx,:)=VRmin+(rand(1,D).*(VRmax-VRmin));
            end

            % Clip positions to the defined bounds
            pos(xpltidx, :) = max(min(pos(xpltidx, :), VRmax), VRmin);
            pos(xplridx, :) = max(min(pos(xplridx, :), VRmax), VRmin);        %---------------------------------------------------------------
        end
        out = benchmark(pos,sfn,0);                                %%%%% Evluate Function based on new position
        out1 = out(1:ps/2); out2 = out(ps/2+1:ps);
        [~, sidx] = sort(out1);
        wnew=WI(sidx);            
        %-------------------Update Personal Bests--------------------------
     pidx = pbestval >= out;  % Create logical index
pbest(pidx, :) = pos(pidx, :);  % Use logical indexing
pbestval(pidx) = out(pidx);

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
