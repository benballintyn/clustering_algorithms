function [centroids,classifications] = mean_shift(data,nCandidates,r,varargin)
p = inputParser;
isIntCheck = @(x) mod(x,1) == 0;
addRequired(p,'data',@isnumeric)
addRequired(p,'nCandidates',isIntCheck)
addRequired(p,'r',@isnumeric)
addParameter(p,'UsePlot',false,@islogical)
addParameter(p,'PlotPauseTime',.5,@isnumeric)
parse(p,data,nCandidates,r,varargin{:})

ndims = size(data,2);
maxs = max(data,[],1);
mins = min(data,[],1);

% Initialize candidate centroids
candidates = (maxs - mins) .* rand(nCandidates,ndims) + mins;

% If using plotting, plot data and initial candidates
if (p.Results.UsePlot)
    h = figure;
    set(h,'Position',[10 10 1400 1600])
    if (ndims == 1)
        p1 = scatter(data,ones(1,length(data)),20,'k.');
        hold on;
        p2 = scatter(candidates,ones(1,nclusters),100,'k.');
    elseif (ndims == 2)
        p1 = scatter(data(:,1),data(:,2),20,'k.');
        hold on;
        p2 = scatter(candidates(:,1),candidates(:,2),100,'r.');
    elseif (ndims == 3)
        p1 = scatter3(data(:,1),data(:,2),data(:,3),20,'k.');
        hold on;
        p2 = scatter3(candidates(:,1),candidates(:,2),candidates(:,3),100,'r.');
    else
        [COEFF] = pca(data);
        %[~, SCORE_MEANS] = pca(means);
        p1 = scatter3(data*COEFF(:,1),data*COEFF(:,2),data*COEFF(:,3),20,'k.');
        hold on;
        p2 = scatter3(candidates*COEFF(:,1),candidates*COEFF(:,2),candidates*COEFF(:,3),100,'r.');
    end
    pause(p.Results.PlotPauseTime)
end

% Set convergence threshold based on variability of the data
thresh = min(vecnorm(data))*1e-4;
d_candidates = realmax*ones(1,nCandidates);

% Until convergence, move candidate centroids towards the mean of the
% points within their radius
nIterations = 1;
while (max(d_candidates > thresh))
    D = pdist2(data,candidates);
    new_candidates = zeros(nCandidates,ndims);
    for i=1:nCandidates
        withinRadiusInds = D(:,i) < r;
        if (~any(withinRadiusInds))
            new_candidates(i,:) = candidates(i,:);
        else
            new_candidates(i,:) = mean(data(withinRadiusInds,:));
        end
    end
    if (p.Results.UsePlot)
        if (ndims == 1)
            p2.XData = new_candidates;
        elseif (ndims == 2)
            p2.XData = new_candidates(:,1);
            p2.YData = new_candidates(:,2);
        elseif (ndims == 3)
            p2.XData = new_candidates(:,1);
            p2.YData = new_candidates(:,2);
            p2.ZData = new_candidates(:,3);
        else
            p2.XData = new_candidates*COEFF(:,1);
            p2.YData = new_candidates*COEFF(:,2);
            p2.ZData = new_candidates*COEFF(:,3);
        end
        h.Children.Title.String = ['Iteration # ' num2str(nIterations)];
        pause(p.Results.PlotPauseTime)
    end
    d_candidates = sqrt(sum((new_candidates - candidates).^2,2));
    candidates = new_candidates;
    nIterations = nIterations + 1;
end

% Post-processing. Remove overlapping candidates
D = pdist2(data,candidates);
[~,classifications] = min(D,[],2);

D = dist(candidates');
rmList = [];
keepList = [];
for i=1:nCandidates
    if (ismember(i,rmList))
        continue;
    else
        rmPossibles = find(D(i,setdiff(1:nCandidates,i)) < r);
        rmList = [rmList setdiff(rmList,keepList)];
        if (sum(classifications == i) > 0)
            keepList = [keepList i];
        else
            rmList = [rmList i];
        end
    end
end

centroids = candidates(keepList,:);
D = pdist2(data,centroids);
[~,classifications] = min(D,[],2);
end

