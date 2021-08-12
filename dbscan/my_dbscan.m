function [clusters] = my_dbscan(data,epsilon,minPoints,varargin)
p = inputParser;
isIntCheck = @(x) mod(x,1) == 0;
addRequired(p,'data',@isnumeric)
addRequired(p,'epsilon',@isnumeric)
addRequired(p,'minPoints',isIntCheck);
parse(p,data,epsilon,minPoints,varargin{:})

npoints = size(data,1);
unvisited = 1:npoints;
visited = [];
nClusters = 0;
clusters = {};

while (~isempty(unvisited))
    randInd = unvisited(ceil(rand*length(unvisited)));
    nClusters = nClusters + 1;
    clusters{nClusters} = [];
    clusters{nClusters} = [clusters{nClusters} randInd];
    visited = [visited randInd];
    
    unvisited = setdiff(unvisited,randInd);
    disp([num2str((length(unvisited)/npoints)*100) ' % remaining'])
    D = pdist2(data(unvisited,:),data(randInd,:));
    neighborhood = find(D < epsilon)';
    
    while (~isempty(neighborhood))
        curInd = neighborhood(1);
        clusters{nClusters} = [clusters{nClusters} curInd];
        visited = [visited curInd];
        unvisited = setdiff(unvisited,curInd);
        disp([num2str((length(unvisited)/npoints)*100) ' % remaining'])
        
        neighborhood = setdiff(neighborhood,curInd);
        
        otherInds = setdiff(1:npoints,[visited neighborhood]);
        D = pdist2(data(otherInds,:),data(curInd,:));
        
        neighborhood = [neighborhood otherInds(find(D < epsilon))];
    end
end

end

