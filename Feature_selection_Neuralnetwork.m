% This code represents some feature selection algorithms in matlab
% Reference: Introduction to Feature Selection, Mathworks
% See: https://kr.mathworks.com/help/deeplearning/ug/choose-a-multilayer-neural-network-training-function.html
% 21. 04. 27, MS Chang, Sogang univ, Korea

clear all; close all; clc;
rng default

% Changable variables, C: # of features, hiddenLayerSize: nodes and layers of ANN
global C 

%% Load dataset

load ionosphere % load the sample data (X: predictor variables, Y: Response variable)

n = randperm(length(X)); X = X(n,:); Y = Y(n);  % Data shuffle
C = 10; % Number of features

%% [Filter type EX 1] ==================== FSCCHI2 ==============
% Univariate feature ranking for classification using chi-square tests

[chi.idx, chi.scores] = fscchi2(X,Y); % Rank the predictors using chi-square tests

% The values in 'scores' are the negative logs of the p-values
% If a p-value is too small, the corresponding score value is Inf

find(isinf(chi.scores)) % 'scores' does not include Inf values
% If scores includes Inf values, you can replace Inf by a larg numeric number

figure('color','w')
subplot(331);
bar(chi.scores(chi.idx)); hold on;
xlabel('Predictor rank'); ylabel('Predictor importance score'); legend('Fscchi2')

% Plot the infinite importances
chi.idxInf = find(isinf(chi.scores)), bar(chi.scores(chi.idx(length(chi.idxInf)+1))*ones(length(chi.idxInf),1))

FSresult.chi = [chi.idx', chi.scores(chi.idx)'];

%% [Filter type EX 2] ==================== FSCMRMR ==============
% Rank features for classification using minimum redundancy maximum relevance algorithm

[mrmr.idx, mrmr.scores] = fscmrmr(X,Y);

subplot(332);
bar(mrmr.scores(mrmr.idx)); hold on;
xlabel('Predictor rank'); ylabel('Predictor importance score'); legend('Fscmrmr')

% Plot the infinite importances
mrmr.idxInf = find(isinf(mrmr.scores)), bar(mrmr.scores(mrmr.idx(length(mrmr.idxInf)+1))*ones(length(mrmr.idxInf),1))

FSresult.mrmr = [mrmr.idx', mrmr.scores(mrmr.idx)'];

%% [Filter type EX 3] ==================== FSRFTEST ==============
% Univariate feature ranking for regression using F-tests
[f_test.idx, f_test.scores] = fsrftest(X,double(cell2mat(Y)));

subplot(333);
bar(f_test.scores(f_test.idx)); hold on;
xlabel('Predictor rank'); ylabel('Predictor importance score'); legend('FSRFTEST')

% Plot the infinite importances
f_test.idxInf = find(isinf(f_test.scores)), bar(f_test.scores(f_test.idx(length(f_test.idxInf)+1))*ones(length(f_test.idxInf),1))

FSresult.f_test = [f_test.idx', f_test.scores(f_test.idx)'];

%% [Filter type EX 4] ==================== Relieff1 ==============
% Rank importance of predictors using ReliefF algorithm
[reli.idx, reli.scores] = relieff(X,double(cell2mat(Y)),10);

subplot(334);
bar(reli.scores(reli.idx)); hold on;
xlabel('Predictor rank'); ylabel('Predictor importance score'); legend('Relieff')


% Plot the infinite importances
reli.idxInf = find(isinf(reli.scores)), bar(reli.scores(reli.idx(length(reli.idxInf)+1))*ones(length(reli.idxInf),1))
FSresult.f_test = [reli.idx', reli.scores(reli.idx)'];

%% [Embedded type EX 1] ================== fitcensemble ==============
t = templateTree('Reproducible',true); % For reproducibiliy of random predictor selections
Mdl = fitcensemble(X,double(cell2mat(Y)),'Method','Bag','NumLearningCycles',50,'Learners',t);
fitcen.scores = oobPermutedPredictorImportance(Mdl);
[~, fitcen.idx] = sort(fitcen.scores,'descend');

subplot(335);
bar(fitcen.scores(fitcen.idx));
xlabel('Predictor rank'); ylabel('Predictor importance score'); legend('fitcen')

%% Feature selection result
drawnow;
FSresult.idx = [chi.idx', mrmr.idx', f_test.idx', reli.idx', fitcen.idx'];

%% Devide training and testset

dataset_origin = [X, double(cell2mat(Y))]; % all inputs
Training_x = dataset_origin(1:round(length(dataset_origin)*0.7),1:end-1);
Training_y = dataset_origin(1:round(length(dataset_origin)*0.7),end);
Test = dataset_origin(round(length(dataset_origin)*0.7)+1:end,1:end-1);

dataset1 = [X(:,chi.idx(1:C)), double(cell2mat(Y))];  % features from chi
Training1_x = dataset1(1:round(length(dataset1)*0.7),1:end-1);
Training1_y = dataset1(1:round(length(dataset1)*0.7),end);
Test1 = dataset1(round(length(dataset1)*0.7)+1:end,1:end-1);

dataset2 = [X(:,mrmr.idx(:,1:C)), double(cell2mat(Y))]; % features from mrmr
Training2_x = dataset2(1:round(length(dataset2)*0.7),1:end-1);
Training2_y = dataset2(1:round(length(dataset2)*0.7),end);
Test2 = dataset2(round(length(dataset2)*0.7)+1:end,1:end-1);

dataset3 = [X(:,f_test.idx(:,1:C)), double(cell2mat(Y))]; % features from ftest
Training3_x = dataset3(1:round(length(dataset3)*0.7),1:end-1);
Training3_y = dataset3(1:round(length(dataset3)*0.7),end);
Test3 = dataset3(round(length(dataset3)*0.7)+1:end,1:end-1);

dataset4 = [X(:,reli.idx(:,1:C)), double(cell2mat(Y))]; % features from relieff
Training4_x = dataset4(1:round(length(dataset4)*0.7),1:end-1);
Training4_y = dataset4(1:round(length(dataset4)*0.7),end);
Test4 = dataset4(round(length(dataset4)*0.7)+1:end,1:end-1);

dataset5 = [X(:,fitcen.idx(:,1:C)), double(cell2mat(Y))]; % features from fitcen
Training5_x = dataset5(1:round(length(dataset5)*0.7),1:end-1);
Training5_y = dataset5(1:round(length(dataset5)*0.7),end);
Test5 = dataset5(round(length(dataset5)*0.7)+1:end,1:end-1);

ANSWER = dataset_origin(round(length(dataset_origin)*0.7)+1:end,end);


% One-hot encoding
N = length(Training_y); 
yn = zeros(N,2); yn1 = yn;  yn2 = yn;  yn3 = yn;  
yn4 = yn; yn5 = yn; yn6 = yn; yn7 = yn;
j = 1; j1 = 1; j2= 1; j3= 1; j4 = 1; j5 = 1;
for k=1:length(ANSWER)
    if Training_y(k) == 98,           yn(j,1) = 1;  else,       yn(j,2) = 1;  end;  j =  j+1;
    if Training1_y(k) == 98,        yn1(j1,1) = 1;  else,     yn1(j1,2) = 1;  end; j1 = j1+1; 
    if Training2_y(k) == 98,        yn2(j2,1) = 1;  else,     yn2(j2,2) = 1;  end; j2 = j2+1;
    if Training3_y(k) == 98,        yn3(j3,1) = 1;  else,     yn3(j3,2) = 1;  end; j3 = j3+1;
    if Training4_y(k) == 98,        yn4(j4,1) = 1;  else,     yn4(j4,2) = 1;  end; j4 = j4+1;
    if Training5_y(k) == 98,        yn5(j5,1) = 1;  else,     yn5(j5,2) = 1;  end; j5 = j5+1;
end
ANSWERn = zeros(length(ANSWER),2); jj = 1; 
for k=1:length(ANSWER)
    if ANSWER(k) == 98, ANSWERn(jj,1) = 1;  else, ANSWERn(jj,2) = 1;  end; jj = jj+1;
end

%% Model Training (Ensemble learning - Bagging tree)

[~,net] = NN(Training_x,yn);   % all inputs
[~,net1] = NN(Training1_x,yn1); % features from chi
[~,net2] = NN(Training2_x,yn2); % features from mrmr
[~,net3] = NN(Training3_x,yn3); % features from ftest
[~,net4] = NN(Training4_x,yn4); % features from relieff5
[~,net5] = NN(Training5_x,yn5); % features from fitcen


%% Test
y =  round(net(Test')');     % all inputs
y1 = round(net1(Test1')');  % features from chi
y2 = round(net2(Test2')');  % features from mrmr
y3 = round(net3(Test3')');  % features from ftest
y4 = round(net4(Test4')');  % features from relieff
y5 = round(net5(Test5')');  % features from fitcen


% Confusion matrix
figure('color','w'); plotconfusion(ANSWERn',y');  title('All input') 
figure('color','w'); plotconfusion(ANSWERn',y1'); title('Chi input')
figure('color','w'); plotconfusion(ANSWERn',y2'); title('MRMR input')
figure('color','w'); plotconfusion(ANSWERn',y3'); title('Ftest input')
figure('color','w'); plotconfusion(ANSWERn',y4'); title('relieff5 input')
figure('color','w'); plotconfusion(ANSWERn',y5'); title('fitcen input')

classes = [1 2];
ANSWERnn = onehotdecode(ANSWERn,classes,2,'single');
ynn = onehotdecode(y,classes,2,'single');
ynn1 = onehotdecode(y1,classes,2,'single');
ynn2 = onehotdecode(y2,classes,2,'single');
ynn3 = onehotdecode(y3,classes,2,'single');
ynn4 = onehotdecode(y4,classes,2,'single');
ynn5 = onehotdecode(y5,classes,2,'single');

N = length(Test);

% Accuracy
Accuracy = {'All input',  sum(trace(confusionmat(ANSWERnn,ynn)))/N*100;
            'Chi-square', sum(trace(confusionmat(ANSWERnn,ynn1)))/N*100;
            'MRMR',       sum(trace(confusionmat(ANSWERnn,ynn2)))/N*100;
            'F-test',     sum(trace(confusionmat(ANSWERnn,ynn3)))/N*100;
            'Relieff5',   sum(trace(confusionmat(ANSWERnn,ynn4)))/N*100;
            'fitcen',     sum(trace(confusionmat(ANSWERnn,ynn5)))/N*100;
            }
        
%% [Appendix] Functions
function [y,net] = NN(input,output)

x = input';
t = output';

hiddenLayerSize = [100 100 100]; %%%%% 

trainFcn = 'trainscg';  % Scaled conjugate gradient backpropagation. (% trainrp trainscg)

% Create a Pattern Recognition Network
net = patternnet(hiddenLayerSize, trainFcn);
net = init(net); %%%%%
net.layers{1}.transferFcn = 'logsig'; %%%%% 
net.layers{2}.transferFcn = 'logsig'; %%%%%
net.layers{3}.transferFcn = 'logsig'; %%%%%

% Choose Input and Output Pre/Post-Processing Functions
% For a list of all processing functions type: help nnprocess
net.input.processFcns = {'mapstd', 'processpca', 'removeconstantrows', 'fixunknowns'}; %%%%%
% mapminmax mapstd processpca fixunknowns removeconstantrows

% Setup Division of Data for Training, Validation, Testing
% For a list of all data division functions type: help nndivision
net.divideFcn = 'dividerand';  % Divide data randomly
net.divideMode = 'sample';  % Divide up every sample
net.divideParam.trainRatio = 90/100;
net.divideParam.valRatio = 10/100;
net.divideParam.testRatio = 0/100;

% Choose a Performance Function
% For a list of all performance functions type: help nnperformance
net.performFcn = 'crossentropy';  % Cross-Entropy

% Choose Plot Functions
% For a list of all plot functions type: help nnplot
net.plotFcns = {'plotperform','plottrainstate','ploterrhist', ...
    'plotconfusion', 'plotroc'};

% Train the Network
[net,tr] = train(net,x,t);

% Test the Network
y = net(x);

% Deployment
% Change the (false) values to (true) to enable the following code blocks.
% See the help for each generation function for more information.
if (false)
    genFunction(net,'myNeuralNetworkFunction');
    y = myNeuralNetworkFunction(x);
end
if (false)
    genFunction(net,'myNeuralNetworkFunction','MatrixOnly','yes');
    y = myNeuralNetworkFunction(x);
end
if (false)
    gensim(net);
end
end
