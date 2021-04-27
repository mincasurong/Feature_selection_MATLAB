% This code represents some feature selection algorithms in matlab
% Reference: shorturl.at/quBLT
% 21. 04. 27, MS Chang, Sogang univ, Korea

clear all; close all; clc;
rng default
%% Load dataset

load ionosphere % load the sample data (X: predictor variables, Y: Response variable)

n = randperm(length(X)); X = X(n,:); Y = Y(n);  % Data shuffle

%% [Filter type EX 1] FSCCHI2
% Univariate feature ranking for classification using chi-square tests

[chi.idx, chi.scores] = fscchi2(X,Y); % Rank the predictors using chi-square tests

% The values in 'scores' are the negative logs of the p-values
% If a p-value is too small, the corresponding score value is Inf

find(isinf(chi.scores)) % 'scores' does not include Inf values
% If scores includes Inf values, you can replace Inf by a larg numeric number

figure('color','w')
bar(chi.scores(chi.idx)); hold on;
xlabel('Predictor rank'); ylabel('Predictor importance score'); title('Fscchi2')

% Plot the infinite importances
chi.idxInf = find(isinf(chi.scores)), bar(chi.scores(chi.idx(length(chi.idxInf)+1))*ones(length(chi.idxInf),1))

FSresult.chi = [chi.idx', chi.scores(chi.idx)'];

%% [Filter type EX 2] FSCMRMR
% Rank features for classification using minimum redundancy maximum relevance algorithm

[mrmr.idx, mrmr.scores] = fscmrmr(X,Y);

figure('color','w')
bar(mrmr.scores(mrmr.idx)); hold on;
xlabel('Predictor rank'); ylabel('Predictor importance score'); title('Fscmrmr')

FSresult.mrmr = [mrmr.idx', mrmr.scores(mrmr.idx)'];
FSresult.idx = [chi.idx', mrmr.idx'];

figure('color','w')
bar(chi.scores/max(chi.scores)); hold on; 
bar(mrmr.scores/max(mrmr.scores));
legend('Chi','MRMR');

%% Training and testset

dataset_origin = [X, double(cell2mat(Y))]; % original
Training = dataset_origin(1:round(length(dataset_origin)*0.7),:);
Test = dataset_origin(round(length(dataset_origin)*0.7)+1:end,1:end-1);

dataset1 = [X(:,chi.idx(1:10)), double(cell2mat(Y))];  % features from chi
Training1 = dataset1(1:round(length(dataset1)*0.7),:);
Test1 = dataset1(round(length(dataset1)*0.7)+1:end,1:end-1);


dataset2 = [X(:,mrmr.idx(:,1:10)), double(cell2mat(Y))]; % features from mrmr
Training2 = dataset1(1:round(length(dataset2)*0.7),:);
Test2 = dataset2(round(length(dataset2)*0.7)+1:end,1:end-1);

%% Training (Ensemble learning - Bagging tree)
[trainedClassifier, validationAccuracy] = trainClassifier(Training);
[trainedClassifier1, validationAccuracy1] = trainClassifier1(Training1);
[trainedClassifier2, validationAccuracy2] = trainClassifier2(Training2);

%% Test
ANSWER = dataset_origin(round(length(dataset_origin)*0.7)+1:end,end);
y = trainedClassifier.predictFcn(Test);
y1 = trainedClassifier1.predictFcn(Test1);
y2 = trainedClassifier2.predictFcn(Test2);

% One-hot encoding
N = length(ANSWER);
ANSWERn = zeros(N,2); yn = ANSWERn; yn1 = yn; yn2 =yn; 
jj = 1; j = 1; j1 = 1; j2= 1;
for k=1:length(ANSWER)
    if ANSWER(k) == 98, ANSWERn(j,1) = 1;  else, ANSWERn(jj,2) = 1;  end; jj = jj+1;
    if y(k) == 98, yn(j,1) = 1; else, yn(j,2) = 1; end; j = j+1;
    if y1(k) == 98, yn1(j1,1) = 1; else, yn1(j1,2) = 1; end; j1 = j1+1; 
    if y2(k) == 98, yn2(j2,1) = 1;  else, yn2(j2,2) = 1; end; j2 = j2+1;
end
figure('color','w'); plotconfusion(ANSWERn',yn'); title('All input')
figure('color','w'); plotconfusion(ANSWERn',yn1'); title('Chi input')
figure('color','w'); plotconfusion(ANSWERn',yn2'); title('MRMR input')

%% Function
function [trainedClassifier, validationAccuracy] = trainClassifier(trainingData)
inputTable = array2table(trainingData, 'VariableNames', {'column_1', 'column_2', 'column_3', 'column_4', 'column_5', 'column_6', 'column_7', 'column_8', 'column_9', 'column_10', 'column_11', 'column_12', 'column_13', 'column_14', 'column_15', 'column_16', 'column_17', 'column_18', 'column_19', 'column_20', 'column_21', 'column_22', 'column_23', 'column_24', 'column_25', 'column_26', 'column_27', 'column_28', 'column_29', 'column_30', 'column_31', 'column_32', 'column_33', 'column_34', 'column_35'});

predictorNames = {'column_1', 'column_2', 'column_3', 'column_4', 'column_5', 'column_6', 'column_7', 'column_8', 'column_9', 'column_10', 'column_11', 'column_12', 'column_13', 'column_14', 'column_15', 'column_16', 'column_17', 'column_18', 'column_19', 'column_20', 'column_21', 'column_22', 'column_23', 'column_24', 'column_25', 'column_26', 'column_27', 'column_28', 'column_29', 'column_30', 'column_31', 'column_32', 'column_33', 'column_34'};
predictors = inputTable(:, predictorNames);
response = inputTable.column_35;
isCategoricalPredictor = [false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false];
template = templateTree(...
    'MaxNumSplits', 245);
classificationEnsemble = fitcensemble(...
    predictors, ...
    response, ...
    'Method', 'Bag', ...
    'NumLearningCycles', 30, ...
    'Learners', template, ...
    'ClassNames', [98; 103]);

predictorExtractionFcn = @(x) array2table(x, 'VariableNames', predictorNames);
ensemblePredictFcn = @(x) predict(classificationEnsemble, x);
trainedClassifier.predictFcn = @(x) ensemblePredictFcn(predictorExtractionFcn(x));

trainedClassifier.ClassificationEnsemble = classificationEnsemble;
inputTable = array2table(trainingData, 'VariableNames', {'column_1', 'column_2', 'column_3', 'column_4', 'column_5', 'column_6', 'column_7', 'column_8', 'column_9', 'column_10', 'column_11', 'column_12', 'column_13', 'column_14', 'column_15', 'column_16', 'column_17', 'column_18', 'column_19', 'column_20', 'column_21', 'column_22', 'column_23', 'column_24', 'column_25', 'column_26', 'column_27', 'column_28', 'column_29', 'column_30', 'column_31', 'column_32', 'column_33', 'column_34', 'column_35'});

predictorNames = {'column_1', 'column_2', 'column_3', 'column_4', 'column_5', 'column_6', 'column_7', 'column_8', 'column_9', 'column_10', 'column_11', 'column_12', 'column_13', 'column_14', 'column_15', 'column_16', 'column_17', 'column_18', 'column_19', 'column_20', 'column_21', 'column_22', 'column_23', 'column_24', 'column_25', 'column_26', 'column_27', 'column_28', 'column_29', 'column_30', 'column_31', 'column_32', 'column_33', 'column_34'};
predictors = inputTable(:, predictorNames);
response = inputTable.column_35;
isCategoricalPredictor = [false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false];
partitionedModel = crossval(trainedClassifier.ClassificationEnsemble, 'KFold', 5);
[validationPredictions, validationScores] = kfoldPredict(partitionedModel);
validationAccuracy = 1 - kfoldLoss(partitionedModel, 'LossFun', 'ClassifError');
end % All Input
function [trainedClassifier, validationAccuracy] = trainClassifier1(trainingData)
inputTable = array2table(trainingData, 'VariableNames', {'column_1', 'column_2', 'column_3', 'column_4', 'column_5', 'column_6', 'column_7', 'column_8', 'column_9', 'column_10', 'column_11'});

predictorNames = {'column_1', 'column_2', 'column_3', 'column_4', 'column_5', 'column_6', 'column_7', 'column_8', 'column_9', 'column_10'};
predictors = inputTable(:, predictorNames);
response = inputTable.column_11;
isCategoricalPredictor = [false, false, false, false, false, false, false, false, false, false];
template = templateTree(...
    'MaxNumSplits', 245);
classificationEnsemble = fitcensemble(...
    predictors, ...
    response, ...
    'Method', 'Bag', ...
    'NumLearningCycles', 30, ...
    'Learners', template, ...
    'ClassNames', [98; 103]);
predictorExtractionFcn = @(x) array2table(x, 'VariableNames', predictorNames);
ensemblePredictFcn = @(x) predict(classificationEnsemble, x);
trainedClassifier.predictFcn = @(x) ensemblePredictFcn(predictorExtractionFcn(x));
trainedClassifier.ClassificationEnsemble = classificationEnsemble;
trainedClassifier.About = '이 구조체는 분류 학습기 R2020b에서 내보낸 훈련된 모델입니다.';
trainedClassifier.HowToPredict = sprintf('새 예측 변수 열 행렬 X를 사용하여 예측하려면 다음을 사용하십시오. \n yfit = c.predictFcn(X) \n여기서 ''c''를 이 구조체를 나타내는 변수의 이름(예: ''trainedModel'')으로 바꾸십시오. \n \n이 모델은 10개의 예측 변수를 사용하여 훈련되었으므로 X는 정확히 10개의 열을 포함해야 합니다. \nX는 훈련 데이터와 정확히 동일한 순서와 형식의 예측 변수 열만 포함해야 \n합니다. 응답 변수 열이나 앱으로 가져오지 않은 열은 포함시키지 마십시오. \n \n자세한 내용은 <a href="matlab:helpview(fullfile(docroot, ''stats'', ''stats.map''), ''appclassification_exportmodeltoworkspace'')">How to predict using an exported model</a>을(를) 참조하십시오.');
inputTable = array2table(trainingData, 'VariableNames', {'column_1', 'column_2', 'column_3', 'column_4', 'column_5', 'column_6', 'column_7', 'column_8', 'column_9', 'column_10', 'column_11'});

predictorNames = {'column_1', 'column_2', 'column_3', 'column_4', 'column_5', 'column_6', 'column_7', 'column_8', 'column_9', 'column_10'};
predictors = inputTable(:, predictorNames);
response = inputTable.column_11;
isCategoricalPredictor = [false, false, false, false, false, false, false, false, false, false];
partitionedModel = crossval(trainedClassifier.ClassificationEnsemble, 'KFold', 5);
[validationPredictions, validationScores] = kfoldPredict(partitionedModel);
validationAccuracy = 1 - kfoldLoss(partitionedModel, 'LossFun', 'ClassifError');
end % Feature from chi
function [trainedClassifier, validationAccuracy] = trainClassifier2(trainingData)
inputTable = array2table(trainingData, 'VariableNames', {'column_1', 'column_2', 'column_3', 'column_4', 'column_5', 'column_6', 'column_7', 'column_8', 'column_9', 'column_10', 'column_11'});

predictorNames = {'column_1', 'column_2', 'column_3', 'column_4', 'column_5', 'column_6', 'column_7', 'column_8', 'column_9', 'column_10'};
predictors = inputTable(:, predictorNames);
response = inputTable.column_11;
isCategoricalPredictor = [false, false, false, false, false, false, false, false, false, false];
template = templateTree(...
    'MaxNumSplits', 245);
classificationEnsemble = fitcensemble(...
    predictors, ...
    response, ...
    'Method', 'Bag', ...
    'NumLearningCycles', 30, ...
    'Learners', template, ...
    'ClassNames', [98; 103]);
predictorExtractionFcn = @(x) array2table(x, 'VariableNames', predictorNames);
ensemblePredictFcn = @(x) predict(classificationEnsemble, x);
trainedClassifier.predictFcn = @(x) ensemblePredictFcn(predictorExtractionFcn(x));
trainedClassifier.ClassificationEnsemble = classificationEnsemble;
trainedClassifier.About = '이 구조체는 분류 학습기 R2020b에서 내보낸 훈련된 모델입니다.';
trainedClassifier.HowToPredict = sprintf('새 예측 변수 열 행렬 X를 사용하여 예측하려면 다음을 사용하십시오. \n yfit = c.predictFcn(X) \n여기서 ''c''를 이 구조체를 나타내는 변수의 이름(예: ''trainedModel'')으로 바꾸십시오. \n \n이 모델은 10개의 예측 변수를 사용하여 훈련되었으므로 X는 정확히 10개의 열을 포함해야 합니다. \nX는 훈련 데이터와 정확히 동일한 순서와 형식의 예측 변수 열만 포함해야 \n합니다. 응답 변수 열이나 앱으로 가져오지 않은 열은 포함시키지 마십시오. \n \n자세한 내용은 <a href="matlab:helpview(fullfile(docroot, ''stats'', ''stats.map''), ''appclassification_exportmodeltoworkspace'')">How to predict using an exported model</a>을(를) 참조하십시오.');
inputTable = array2table(trainingData, 'VariableNames', {'column_1', 'column_2', 'column_3', 'column_4', 'column_5', 'column_6', 'column_7', 'column_8', 'column_9', 'column_10', 'column_11'});

predictorNames = {'column_1', 'column_2', 'column_3', 'column_4', 'column_5', 'column_6', 'column_7', 'column_8', 'column_9', 'column_10'};
predictors = inputTable(:, predictorNames);
response = inputTable.column_11;
isCategoricalPredictor = [false, false, false, false, false, false, false, false, false, false];
partitionedModel = crossval(trainedClassifier.ClassificationEnsemble, 'KFold', 5);
[validationPredictions, validationScores] = kfoldPredict(partitionedModel);
validationAccuracy = 1 - kfoldLoss(partitionedModel, 'LossFun', 'ClassifError');
end % Feature from mrmr