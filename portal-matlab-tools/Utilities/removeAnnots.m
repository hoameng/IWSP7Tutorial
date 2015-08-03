function removeAnnots(datasets,exp)
% Usage: removeAnnots(IEEGDataset,regexp)
% Function will remove all annotation layers matching regexp exp from
% IEEGDatasets after confirmation
%CAUTION: WILL PERMANENTLY REMOVE 
%v2 7/28/2015 - Updated to support ieeg-matlab-1.13.2


for i = 1:numel(datasets)
    fprintf('Removing layers from %s \n',datasets(i).snapName);
    try
    layers = [datasets(i).annLayer];
    layerNames = {layers.name};
    tmp = cellfun(@(x)regexp(x,exp)>0,layerNames,'UniformOutput',0);
    tmp = cellfun(@(x)(~isempty(x)),tmp);
    layerIdxs = find(tmp~=0);
        for j = layerIdxs
            resp = input(sprintf('Remove layer %s ...? (y/n): ',layerNames{j}),'s');
            if strcmp(resp,'y')
                try
                    datasets(i).removeAnnLayer(layerNames{j});
                    fprintf('...done!\n');
                catch
                    fprintf('...fail!\n');
                end
            end
        end
    catch
        fprintf('No layers found\n');
    end
    
end