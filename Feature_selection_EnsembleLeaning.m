% This code represents some feature selection algorithms in matlab
% Reference: shorturl.at/quBLT
% 21. 04. 27, MS Chang, Sogang univ, Korea

clear all; close all; clc;
rng default
%% Load dataset

load ionosphere % load the sample data (X: predictor variables, Y: Response variable)

n = randperm(length(X)); X = X(n,:); Y = Y(n);  % Data shuffle

%% [Filter type EX 1] ==================== FSCCHI2 ==============
% Univariate feature ranking for classification using chi-square tests

[chi.idx, chi.scores] = fscchi2(X,Y); % Rank the predictors using chi-square tests

% The values in 'scores' are the negative logs of the p-values
% If a p-value is too small, the corresponding score value is Inf

find(isinf(chi.scores)) % 'scores' does not include Inf values
% If scores includes Inf values, you can replace Inf by a larg numeric number

figure('color','w')
subplot(221);
bar(chi.scores(chi.idx)); hold on;
xlabel('Predictor rank'); ylabel('Predictor importance score'); legend('Fscchi2')

% Plot the infinite importances
chi.idxInf = find(isinf(chi.scores)), bar(chi.scores(chi.idx(length(chi.idxInf)+1))*ones(length(chi.idxInf),1))

FSresult.chi = [chi.idx', chi.scores(chi.idx)'];

%% [Filter type EX 2] ==================== FSCMRMR ==============
% Rank features for classification using minimum redundancy maximum relevance algorithm

[mrmr.idx, mrmr.scores] = fscmrmr(X,Y);

subplot(222);
bar(mrmr.scores(mrmr.idx)); hold on;
xlabel('Predictor rank'); ylabel('Predictor importance score'); legend('Fscmrmr')

% Plot the infinite importances
mrmr.idxInf = find(isinf(mrmr.scores)), bar(mrmr.scores(mrmr.idx(length(mrmr.idxInf)+1))*ones(length(mrmr.idxInf),1))

FSresult.mrmr = [mrmr.idx', mrmr.scores(mrmr.idx)'];

%% [Filter type EX 3] ==================== FSRFTEST ==============
% Univariate feature ranking for regression using F-tests
[f_test.idx, f_test.scores] = fsrftest(X,double(cell2mat(Y)));

subplot(223);
bar(f_test.scores(f_test.idx)); hold on;
xlabel('Predictor rank'); ylabel('Predictor importance score'); legend('FSRFTEST')

% Plot the infinite importances
f_test.idxInf = find(isinf(f_test.scores)), bar(f_test.scores(f_test.idx(length(f_test.idxInf)+1))*ones(length(f_test.idxInf),1))

FSresult.f_test = [f_test.idx', f_test.scores(f_test.idx)'];

%% [Filter type EX 4] ==================== Relieff ==============
% Rank importance of predictors using ReliefF algorithm
[reli.idx, reli.scores] = relieff(X,double(cell2mat(Y)),10);

subplot(224);
bar(reli.scores(reli.idx)); hold on;
xlabel('Predictor rank'); ylabel('Predictor importance score'); legend('Relieff')


% Plot the infinite importances
reli.idxInf = find(isinf(reli.scores)), bar(reli.scores(reli.idx(length(reli.idxInf)+1))*ones(length(reli.idxInf),1))
FSresult.f_test = [reli.idx', reli.scores(reli.idx)'];
drawnow;
%% Feature selection result
FSresult.idx = [chi.idx', mrmr.idx', f_test.idx', reli.idx'];

%% Devide training and testset

dataset_origin = [X, double(cell2mat(Y))]; % all inputs
Training = dataset_origin(1:round(length(dataset_origin)*0.7),:);
Test = dataset_origin(round(length(dataset_origin)*0.7)+1:end,1:end-1);

dataset1 = [X(:,chi.idx(1:10)), double(cell2mat(Y))];  % features from chi
Training1 = dataset1(1:round(length(dataset1)*0.7),:);
Test1 = dataset1(round(length(dataset1)*0.7)+1:end,1:end-1);

dataset2 = [X(:,mrmr.idx(:,1:10)), double(cell2mat(Y))]; % features from mrmr
Training2 = dataset2(1:round(length(dataset2)*0.7),:);
Test2 = dataset2(round(length(dataset2)*0.7)+1:end,1:end-1);

dataset3 = [X(:,f_test.idx(:,1:10)), double(cell2mat(Y))]; % features from ftest
Training3 = dataset3(1:round(length(dataset3)*0.7),:);
Test3 = dataset3(round(length(dataset3)*0.7)+1:end,1:end-1);

dataset4 = [X(:,reli.idx(:,1:10)), double(cell2mat(Y))]; % features from relieff
Training4 = dataset4(1:round(length(dataset4)*0.7),:);
Test4 = dataset4(round(length(dataset4)*0.7)+1:end,1:end-1);

%% Model Training (Ensemble learning - Bagging tree)
[trainedClassifier, validationAccuracy]   = trainClassifier(Training);   % all inputs
[trainedClassifier1, validationAccuracy1] = trainClassifier1(Training1); % features from chi
[trainedClassifier2, validationAccuracy2] = trainClassifier1(Training2); % features from mrmr
[trainedClassifier3, validationAccuracy3] = trainClassifier1(Training3); % features from ftest
[trainedClassifier4, validationAccuracy4] = trainClassifier1(Training4); % features from relieff

%% Test
ANSWER = dataset_origin(round(length(dataset_origin)*0.7)+1:end,end);
y = trainedClassifier.predictFcn(Test);     % all inputs
y1 = trainedClassifier1.predictFcn(Test1);  % features from chi
y2 = trainedClassifier2.predictFcn(Test2);  % features from mrmr
y3 = trainedClassifier3.predictFcn(Test3);  % features from ftest
y4 = trainedClassifier4.predictFcn(Test4);  % features from relieff

% One-hot encoding
N = length(ANSWER); 
ANSWERn = zeros(N,2); yn = ANSWERn; 
yn1 = yn;  yn2 = yn;  yn3 = yn;  yn4 = yn;
jj = 1; j = 1; j1 = 1; j2= 1; j3= 1; j4 = 1;
for k=1:length(ANSWER)
    if ANSWER(k) == 98, ANSWERn(jj,1) = 1;  else, ANSWERn(jj,2) = 1;  end; jj = jj+1;
    if y(k) == 98,           yn(j,1) = 1;   else,       yn(j,2) = 1;  end;  j =  j+1;
    if y1(k) == 98,        yn1(j1,1) = 1;   else,     yn1(j1,2) = 1;  end; j1 = j1+1; 
    if y2(k) == 98,        yn2(j2,1) = 1;   else,     yn2(j2,2) = 1;  end; j2 = j2+1;
    if y3(k) == 98,        yn3(j3,1) = 1;   else,     yn3(j3,2) = 1;  end; j3 = j3+1;
    if y4(k) == 98,        yn4(j4,1) = 1;   else,     yn4(j4,2) = 1;  end; j4 = j4+1;
end

% Confusion matrix
figure('color','w'); plotconfusion(ANSWERn',yn');  title('All input') 
figure('color','w'); plotconfusion(ANSWERn',yn1'); title('Chi input')
figure('color','w'); plotconfusion(ANSWERn',yn2'); title('MRMR input')
figure('color','w'); plotconfusion(ANSWERn',yn3'); title('Ftest input')
figure('color','w'); plotconfusion(ANSWERn',yn4'); title('relieff input')

% Accuracy
Accuracy = {'All input',  sum(trace(confusionmat(ANSWER,y)))/N*100;
            'Chi-square', sum(trace(confusionmat(ANSWER,y1)))/N*100;
            'MRMR',       sum(trace(confusionmat(ANSWER,y2)))/N*100;
            'F-test',     sum(trace(confusionmat(ANSWER,y3)))/N*100;
            'Relieff',    sum(trace(confusionmat(ANSWER,y4)))/N*100;
            }
        
%% [Appendix] Functions
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
end % Feature