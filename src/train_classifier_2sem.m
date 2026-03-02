function [trainedClassifier, validationAccuracy] = train_classifier_2sem(trainingData, responseData)
% [trainedClassifier, validationAccuracy] = train_classifier_2sem(trainingData,responseData)
% Returns a trained classifier and its accuracy. 
% Use this code to automate training the same model with new academic data.
%
% Input:
% trainingData: A matrix with 14 columns of double data type.
%
% responseData: A vector with the categories of the trainingData. 
%              The length of responseData and the number of rows of 
%              trainingData must be equal.
% 
% Output:
% trainedClassifier: A struct containing the trained classifier. 
% The struct contains various fields with information about the trained 
% classifier.
% 
% validationAccuracy: A double representing the validation accuracy as a 
% percentage.
%
% trainedClassifier.predictFcn: Is the function to make predictions on 
% new data.
%
%      
% Use the code to train the model with new data. To retrain your
% classifier, call the function from the command line with your original
% data or new data as the input arguments trainingData and responseData.
%
% For example, to retrain a classifier trained with the original data set 
% T and response Y, enter:
%   [trainedClassifier, validationAccuracy] = train_classifier_2sem(T, Y)
%
% To make predictions with the returned 'trainedClassifier' on new data T2,
% use
%   [yfit,scores] = trainedClassifier.predictFcn(T2)
%
% T2 must be a matrix containing only the predictor columns used for
% training. 
% 
% Extract predictors and response
% This code processes the data into the right shape for training the model.
% Convert input to table
inputTable = array2table(trainingData, 'VariableNames', ... 
                         {'column_1', 'column_2', 'column_3', ...
                          'column_4', 'column_5', 'column_6', ...
                          'column_7', 'column_8', 'column_9', ...
                          'column_10', 'column_11', 'column_12', ...
                          'column_13', 'column_14'});

predictorNames = {'column_1', 'column_2', 'column_3', 'column_4', ...
                  'column_5', 'column_6', 'column_7', 'column_8', ...
                  'column_9', 'column_10', 'column_11', 'column_12', ...
                  'column_13', 'column_14'};

predictors = inputTable(:, predictorNames);
response = responseData;
isCategoricalPredictor = [false, false, false, false, false, false, ...
                          false, false, false, false, false, false, ...
                          false, false];
classNames = {'No'; 'Si'};
costMatrix = [0 5; 1 0];

% Train a classifier
% This code specifies all the classifier options and trains the classifier.
subspaceDimension = max(1, min(7, width(predictors) - 1));
classificationEnsemble = fitcensemble(...
                                      predictors, ...
                                      response, ...
                                      'Method', 'Subspace', ...
                                      'NumLearningCycles', 30, ...
                                      'Learners', 'knn', ...
                                      'NPredToSample', subspaceDimension, ...
                                      'Cost', costMatrix, ...
                                      'ClassNames', classNames);

% Create the result struct with predict function
predictorExtractionFcn = @(x) array2table(x, 'VariableNames', predictorNames);
ensemblePredictFcn = @(x) predict(classificationEnsemble, x);
trainedClassifier.predictFcn = @(x) ensemblePredictFcn(predictorExtractionFcn(x));

% Add additional fields to the result struct
trainedClassifier.ClassificationEnsemble = classificationEnsemble;
trainedClassifier.About = 'This struct is a trained model exported from MATLAB R2025b.';

% Perform cross-validation
partitionedModel = crossval(trainedClassifier.ClassificationEnsemble, 'KFold', 5);

% Compute validation predictions
[validationPredictions, validationScores] = kfoldPredict(partitionedModel);

% Compute validation accuracy
validationAccuracy = 1 - kfoldLoss(partitionedModel, 'LossFun', 'ClassifError');
