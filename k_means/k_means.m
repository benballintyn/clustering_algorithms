function [means,classifications] = k_means(data,nclusters,varargin)
p = inputParser;
isIntCheck = @(x) mod(x,1) == 0;
addRequired(p,'data',@isnumeric)
addRequired(p,'nclusters',isIntCheck)
addParameter(p,'UsePlot',false,@islogical)
addParameter(p,'EmptyClusterRule','AssignToLonelyPoint',@ischar)
addParameter(p,'AttemptClusterSplit',false,@islogical)
addParameter(p,'ClusterSplitAttempts',1,isIntCheck)
addParameter(p,'PlotPauseTime',.5,@isnumeric)
parse(p,data,nclusters,varargin{:})

ndims = size(data,2);
maxs = max(data,[],1);
mins = min(data,[],1);
means = (maxs - mins) .* rand(nclusters,ndims) - mins;
% Open plot with data and randomly initialized means
if (p.Results.UsePlot)
    h = figure;
    set(h,'Position',[10 10 1400 1600])
    if (ndims == 1)
        p1 = scatter(data,ones(1,length(data)),20,'k.');
        hold on;
        p2 = scatter(means,ones(1,nclusters),100,'k.');
    elseif (ndims == 2)
        p1 = scatter(data(:,1),data(:,2),20,'k.');
        hold on;
        p2 = scatter(means(:,1),means(:,2),100,'r.');
    elseif (ndims == 3)
        p1 = scatter3(data(:,1),data(:,2),data(:,3),20,'k.');
        hold on;
        p2 = scatter3(means(:,1),means(:,2),means(:,3),100,'r.');
    else
        [COEFF] = pca(data);
        %[~, SCORE_MEANS] = pca(means);
        p1 = scatter3(data*COEFF(:,1),data*COEFF(:,2),data*COEFF(:,3),20,'k.');
        hold on;
        p2 = scatter3(means*COEFF(:,1),means*COEFF(:,2),means*COEFF(:,3),100,'r.');
    end
    pause(p.Results.PlotPauseTime)
end

thresh = min(vecnorm(data))*1e-4;
d_means = realmax*ones(1,nclusters);
nIterations = 1;
while (max(d_means) > thresh)
    [new_means,d_means] = moveClusters(data,means,nclusters,ndims,p);
    
    if (p.Results.UsePlot)
        if (ndims == 1)
            p2.XData = new_means;
        elseif (ndims == 2)
            p2.XData = new_means(:,1);
            p2.YData = new_means(:,2);
        elseif (ndims == 3)
            p2.XData = new_means(:,1);
            p2.YData = new_means(:,2);
            p2.ZData = new_means(:,3);
        else
            p2.XData = new_means*COEFF(:,1);
            p2.YData = new_means*COEFF(:,2);
            p2.ZData = new_means*COEFF(:,3);
        end
        h.Children.Title.String = ['Iteration # ' num2str(nIterations)];
        pause(p.Results.PlotPauseTime)
    end
    means = new_means;
    nIterations = nIterations + 1;
end
D = pdist2(data,means);
[~,classifications] = min(D,[],2);

if (p.Results.AttemptClusterSplit)
    for i=1:p.Results.ClusterSplitAttempts
        % Compute the mean intra-cluster distance for each cluster
        D = pdist2(data,means);
        [~,classifications] = min(D,[],2);
        for j=1:nclusters
            inds = classifications == j;
            meanIntraClusterDistance(j) = mean(D(inds,j));
        end
        
        % Identify the cluster with the largest intra-cluster distance
        % We will attempt to split this cluster by stealing a centroid from
        % another cluster
        [maxIntraClusterDistance,maxIntraClusterDistanceInd] = max(meanIntraClusterDistance);
        
        % Compute the distance between all centroids (cluster means) and
        % identify the two that are closest together
        % We will take one of the centroids from that pair and move it near
        % the centroid with the largest intra-cluster distance
        centroidDists = dist(means');
        for j=1:nclusters
            minCentroidDists(j) = min(centroidDists(j,setdiff(1:nclusters,j)));
        end
        [~,moveCentroidInd] = min(minCentroidDists);
        
        % Find the axis of largest variation in the cluster with the
        % largest intra-cluster distance
        [temp_COEFF] = pca(data(maxIntraClusterDistanceInd));
        
        % Assign the centroid from the nearby pair to be a perturbed version
        % of the centroid from the large cluster
        old_means = means;
        means(moveCentroidInd,:) = mean(data(maxIntraClusterDistanceInd,:)) + randn*temp_COEFF(:,1)';
        
        if (p.Results.UsePlot)
            if (ndims == 1)
                p2.XData = means;
            elseif (ndims == 2)
                p2.XData = means(:,1);
                p2.YData = new_means(:,2);
            elseif (ndims == 3)
                p2.XData = means(:,1);
                p2.YData = means(:,2);
                p2.ZData = means(:,3);
            else
                p2.XData = means*COEFF(:,1);
                p2.YData = means*COEFF(:,2);
                p2.ZData = means*COEFF(:,3);
            end
            h.Children.Title.String = ['Iteration # ' num2str(nIterations) ' : Split Attempt # ' num2str(i)];
            pause(p.Results.PlotPauseTime)
        end
        
        d_means = sqrt(sum((means - old_means).^2,2));
        while (max(d_means) > thresh)
            [new_means,d_means] = moveClusters(data,means,nclusters,ndims,p);
            if (p.Results.UsePlot)
                if (ndims == 1)
                    p2.XData = new_means;
                elseif (ndims == 2)
                    p2.XData = new_means(:,1);
                    p2.YData = new_means(:,2);
                elseif (ndims == 3)
                    p2.XData = new_means(:,1);
                    p2.YData = new_means(:,2);
                    p2.ZData = new_means(:,3);
                else
                    p2.XData = new_means*COEFF(:,1);
                    p2.YData = new_means*COEFF(:,2);
                    p2.ZData = new_means*COEFF(:,3);
                end
                h.Children.Title.String = ['Iteration # ' num2str(nIterations) ' : Split Attempt # ' num2str(i)];
            end
            means = new_means;
            nIterations = nIterations + 1;
        end
    end
end

D = pdist2(data,means);
[~,classifications] = min(D,[],2);
end

