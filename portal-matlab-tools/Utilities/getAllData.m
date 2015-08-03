function data = getAllData(dataset,channels)
%   Usage: data = getAllData(dataset,channels)
%   dataset    -    IEEGDataset
%   channels   -    array of channel indices
%   Gets all data in IEEGDataset dataset for specified channels.
%NOTE: If large dataset, use getExtendedData, which gets data one channel
%at a time

blockLen = 3600;
duration = dataset.channels(1).get_tsdetails.getDuration;
fs = dataset.sampleRate;

numPoints = duration/1e6*fs;
numBlocks = ceil(numPoints/blockLen);

data = cell(numBlocks,1);
for i = 1:numBlocks
    startPt = 1+(i-1)*blockLen;
    endPt = i*blockLen;
    try
        data{i} = dataset.getvalues(startPt:min(endPt,numPoints),channels);
    catch
        data{i} = dataset.getvalues(startPt:min(endPt,numPoints),channels);
    end
end
data = cell2mat(data);
