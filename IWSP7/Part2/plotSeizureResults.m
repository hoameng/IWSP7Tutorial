%%% Script to clip interictal, preictal, ictal segments from Dog Dataset
%%% from IEEG Portal for IWSP7 IEEG Tutorial
%%% Note: this will not be shown to course; internal use only

clear; clc;
addpath(genpath('../../ieeg-matlab-1.13.2'));
addpath(genpath('../../portal-matlab-tools/'));

addpath('../../')
% Portal Data Set Options
S2US = 1e6;
US2S = 1e-6;
%ieegUser = 'akhambhati';
%ieegPwd = 'akh_ieeglogin.bin';
ieegUser = 'hoameng';
ieegPwd = 'hoa_ieeglogin.bin';
snapshotName = 'I004_A0003_D001';
snapTrainPrefix = '-TrainingAnnots';
trainAnnLayerName = 'Train';
snapTestPrefix = '-TestingAnnots';
testAnnLayerName = 'Real';


%%% Grab testing snapshot, extract features, predict using model
% Load the Testing snapshot
testSnapName = strcat(snapshotName, snapTestPrefix);
try
    session = IEEGSession(testSnapName, ieegUser, ieegPwd);
catch
    error('You must first create the %s snapshot', testSnapName);
end
dataset = session.data;

layerNames = {dataset.annLayer(:).name};

% Create a matrix
annotTimes = cell(numel(layerNames),1);
annotObjs = cell(numel(layerNames),1);
for i = 1:numel(layerNames)
    % Locate annotations in prescribed layer
    [annotObjs{i},annotTimes{i},~] = getAllAnnots(dataset, layerNames{i});
end

% Get the Real annotTimes to compute accuracy
for i = 1:numel(annotTimes)
    if(strcmp(layerNames{i},'Real'))
        real_annotTimes = annotTimes{i};
        real_annotObjs = annotObjs{i};
    end
end

colors = jet(numel(annotTimes));
figure(1),
hax=axes;
title('IWSP Tutorial Team Results'),
real_i = 1;
for i = 1:numel(annotTimes)
    if(strcmp(layerNames{i},'Real') || strcmp(layerNames{i},'Test'))
        continue;
    else
        line([annotTimes{i}(:,1) annotTimes{i}(:,1)],[real_i-1 + 0.25 real_i-1+0.75],'Color',colors(i,:));
        text(annotTimes{i}(1,1), real_i-1 + 0.8, layerNames{i});
        hold on;
        real_i = real_i + 1;
    end
end

for i = 1:numel(annotTimes)
    if(strcmp(layerNames{i},'Test'))
        line([annotTimes{i}(:,1) annotTimes{i}(:,1)],[real_i-1 + 0.25 real_i-1+0.75],'Color',colors(i,:));
        text(annotTimes{i}(1,1), real_i-1 + 0.8, layerNames{i});
        hold on;
        real_i = real_i + 1;
    elseif(strcmp(layerNames{i},'Real'))
        idx = find(strcmp({annotObjs{i}.type},'SZ'));
        line([annotTimes{i}(idx,1) annotTimes{i}(idx,1)],[real_i-1+0.25 real_i-1+0.75], 'Color', 'k');
        text(annotTimes{i}(1,1), real_i-1 + 0.8, layerNames{i});
        hold on;
        real_i = real_i + 1;
    end
end

ylim([0 numel(annotTimes)+1])

