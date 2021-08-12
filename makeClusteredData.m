function [data] = makeClusteredData(nDataPoints,ndims,nclusters,meanMax,stdMax)
data = zeros(nDataPoints,ndims);
inds = ceil(rand(nDataPoints,1)*nclusters);
data_means = randn(nclusters,ndims)*meanMax;
data_stds = rand(nclusters,ndims)*stdMax;
for i=1:ndims
    for j=1:nclusters
        data(inds == j,i) = normrnd(data_means(j,i),data_stds(j,i),sum(inds == j),1);
    end
end
end

