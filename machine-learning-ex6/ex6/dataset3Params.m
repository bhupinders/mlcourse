function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
C = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
sigma = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];

accuracy = zeros(size(C,2)*size(sigma,2), 3);

for i = 1:size(C,2)
  for k =1:size(sigma,2)
    model= svmTrain(X, y, C(i), @(x1, x2) gaussianKernel(x1, x2, sigma(k)));
    prediction = svmPredict(model, Xval);
    %fprintf('Prediction at sigma = %f : and C : %f is %f', sigma(k), C(i), mean(double(prediction == yval)));
    accuracy(8*(i-1)+k, :) = [sigma(k) C(i) mean(double(prediction == yval))];
  end
 end
    

%model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));

%prediction = svmPredict(model, Xval);

%mean(double(prediction == yval))
%size(accuracy)
[i,p] = max(accuracy, [], 1);
maxIndex = p(3);
maxRow = accuracy(maxIndex,:);
C = maxRow(2);
sigma = maxRow(1);
% =========================================================================

end
