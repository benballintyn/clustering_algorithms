function [new_means,d_means] = moveClusters(data,means,nclusters,ndims,p)
% D is MX x MY matrix of distances where MX = # of data points and
% MY = # of clusters
D = pdist2(data,means);
[~,classifications] = min(D,[],2);
new_means = zeros(nclusters,ndims);
for i = 1:nclusters
    inds = classifications == i;
    if (~any(inds))
        switch p.Results.EmptyClusterRule
            case 'AssignToLonelyPoint'
                minDists = min(D,[],2);
                [~,maxInd] = max(minDists);
                new_means(i,:) = data(maxInd,:);
            case 'AttemptClusterSplit'
                tempD = pdist2(data,means);
                [~,temp_classifications] = min(D,[],2);
                for j=1:nclusters
                    temp_inds = temp_classifications == j;
                    meanIntraClusterDistance(j) = mean(D(temp_inds,j));
                end
                [maxIntraClusterDistance,maxIntraClusterDistanceInd] = max(meanIntraClusterDistance);
                new_means(i,:) = means(maxIntraClusterDistanceInd,:) + rand(1,ndims)*maxIntraClusterDistance/2;
            case 'AttemptClusterSplit_FindPCAxis'
                tempD = pdist2(data,means);
                [~,temp_classifications] = min(D,[],2);
                for j=1:nclusters
                    temp_inds = temp_classifications == j;
                    meanIntraClusterDistance(j) = mean(D(temp_inds,j));
                end
                [maxIntraClusterDistance,maxIntraClusterDistanceInd] = max(meanIntraClusterDistance);
                [temp_COEFF] = pca(data(maxIntraClusterDistanceInd));
                new_means(i,:) = mean(data(maxIntraClusterDistanceInd,:)) + randn*temp_COEFF(:,1)';
            otherwise
                error('EmptyClusterRule not recognized')
        end
    else
        new_means(i,:) = mean(data(inds,:),1);
    end
end
d_means = sqrt(sum((means - new_means).^2,2));
end

