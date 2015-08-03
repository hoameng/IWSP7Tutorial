%%% Script to clip interictal, preictal, ictal segments from Dog Dataset
%%% from IEEG Portal for IWSP7 IEEG Tutorial
%%% This is NOT the optimal framework for seizure detection

clear; clc;
addpath(genpath('../../ieeg-matlab-1.13.2'));
addpath(genpath('../../portal-matlab-tools/'));

addpath('../../')

%% Portal Data Set Options
%
% Empty space to configure IEEG Account Login, IEEG Snapshot, Time Conversion Constants
%
S2US = 1e6;
US2S = 1e-6;
ieegUser = '';
ieegPwd = '';
snapshotName = 'I004_A0003_D001';
snapTrainPrefix = '-TrainingAnnots';
trainAnnLayerName = 'Train';
snapTestPrefix = '-TestingAnnots';
testAnnLayerName = 'Test';

%% Grab training snapshot, extract features, train model
%
% Insert code to instantiate an IEEG Session pointing training data snapshot
%
trainSnapName = strcat(snapshotName, snapTrainPrefix);
session = IEEGSession(trainSnapName, ieegUser, ieegPwd);
dataset = session.data;

%
% Insert code to Retrieve dataset sampling frequency, signal length, channel indices
%
Fs = dataset.sampleRate;
dsDurSn = dataset.rawChannels(1).get_tsdetails.getDuration * US2S * Fs;
dsDurSn = floor(dsDurSn);
channelsIdx = 1:numel(dataset.rawChannels);

%
% Select the "Train" Annotation layer, and grab all associated annotations
% Retrieve and store annotation object for every annotation in the layer
[allAnn, allAnnUsec, allAnnChans] = getAllAnnots(dataset, trainAnnLayerName);

%
% For Machine Learning, initialize matrices
% Matrix for features per clip, and training label per clip (interictal/ictal)
%
featMatr = zeros(numel(allAnn), 1);
trainLabel = zeros(numel(allAnn), 1);

%
% Iterate over each annotation object
% For each annotation:
%    1. Retrieve time range and associated signal clip
%    2. Compute and store features for that clip
%    3. Store the binary label for that clip
%        (Clip label is stored as an annotation type for a given annotation layer)
%        (We use type 'NSZ' for interictal and 'SZ' for ictal)
%
for i = 1:numel(allAnn)
   disp(sprintf('Features from clip %d of %d', i, numel(allAnn)))

   % Get values for each
   snRange = allAnn(i).start * US2S * Fs : allAnn(i).stop * US2S * Fs;
   annData = getExtendedData(dataset, snRange, channelsIdx);

   % Compute features and add to feature matrix
   feat = feat_LineLength(annData);
   featMatr(i, 1) = feat;

   % Save label for each
   if strcmp(allAnn(i).type, 'NSZ')
       trainLabel(i, 1) = 1;
   end
   if strcmp(allAnn(i).type, 'SZ')
       trainLabel(i, 1) = 2;
   end
end

%
% With clip interictal/ictal labels and feature set, train ML algorithm
%
disp(sprintf('\nTraining Logistic Regression\n'))
B = mnrfit(featMatr, trainLabel);

%% Grab testing snapshot, extract features, predict using model, upload prediction
%
% Insert code to instantiate an IEEG Session pointing testing data snapshot
% You may append the snapshot to the current session, rather than restarting the session
%
testSnapName = strcat(snapshotName, snapTestPrefix);
session.openDataSet(testSnapName);

%
% Insert code to Retrieve dataset sampling frequency, signal length, channel indices
%
dataset = session.data(2);
Fs = dataset.sampleRate;
dsDurSn = dataset.rawChannels(1).get_tsdetails.getDuration * US2S * Fs;
dsDurSn = floor(dsDurSn);
channelsIdx = 1:numel(dataset.rawChannels);

%
% Select the "Test" Annotation layer, and grab all associated annotations
% Retrieve and store annotation object for every annotation in the layer
%
[allAnn, allAnnUsec, allAnnChans] = getAllAnnots(dataset, testAnnLayerName);

%
% For Machine Learning, initialize matrices
% Matrix for predicted label per clip
%
predLabel = zeros(numel(allAnn), 1);

%
% Iterate over each annotation object
% For each annotation:
%    1. Retrieve time range and associated signal clip
%    2. Compute features for that clip
%    3. Use ML algorithm to predict clip label (interictal/ictal)
%    4. Store the binary label for that clip
%
for i = 1:numel(allAnn)
   fprintf('Predicting clip %d of %d', i, numel(allAnn))

   % Get values for each
   snRange = allAnn(i).start * US2S * Fs : allAnn(i).stop * US2S * Fs;
   annData = getExtendedData(dataset, snRange, channelsIdx);

   % Compute features
   feat = feat_LineLength(annData);

   % Predict logistic regression
   phat = mnrval(B, feat);
   [~, predLabel(i, 1)] = max(phat);

   if(predLabel(i,1) == 2)
       fprintf('.... Detected seizure !!!\n')
   else
       fprintf('\n')
   end
end

%
% Retrieve the predicted labels that indicate seizures
% Grab the start and stop time for each of the predicted seizure labels 
%   (corresponding to original annotation start/stop time)
% Also, retrieve the channels on which the prediction occurred (in our case all channels)
% Use the uploadAnnotations tool to
%    1. Create an annotation layer with the same name as your ieegUser name
%    2. Post seizure annotations with the annotation type: 'seizure'
%
seizures = allAnnUsec(predLabel(:,1) == 2, :);
channels = allAnnChans(predLabel(:,1)== 2)';
uploadAnnotations(dataset,ieegUser,seizures,channels,'seizure');

%% Delete session
session.delete;
