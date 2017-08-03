function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%


for i = 1:m
  J += ( X(i,:)*theta - y(i,1) ) ^ 2; 
endfor

J = J/(2*m);

for i = 2:rows(theta)
  J += lambda * theta(i,1) ^ 2 / (2*m);
endfor


for i = 1:m
  grad(1,1) += (X(i,:) * theta - y(i,1))* X(i,1) / m;
endfor


for j = 2: rows(theta)
  for i = 1:m
    grad(j,1) += (X(i,:) * theta - y(i,1))* X(i,j) / m; 
  endfor
  grad(j,1) += lambda*theta(j,1)/m;
endfor




% =========================================================================

grad = grad(:);

end
