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

C = [0.01 0.03 0.1 0.3 1 3 10 30];
sigma = [0.01 0.03 0.1 0.3 1 3 10 30];
C_len = length(C(1,:));
sigma_len = length(sigma(1,:));
err = zeros(C_len, sigma_len);

x1 = X(:,1);
x2 = X(:,2);

for i=1:C_len
  for j=1:sigma_len
    my_model = svmTrain(X, y, C(i), @(x1, x2)gaussianKernel(x1, x2, sigma(j)));
    pred = svmPredict(my_model, Xval);
    err(i,j) = mean(double(pred ~= yval));
  end
end

err

[val, minrow] = min(min(err,[],2));
C = C(minrow)

[val, mincol] = min(min(err,[],1));
sigma = sigma(mincol)

% =========================================================================

end

