function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.


for iter = 1:m
%    fprintf('J before = %f\n', J);
    J = J + (X(iter,:) * theta - y(iter,1))^2; 
%    fprintf('J of this turn = %f\n', J);
endfor
J = J/(2*m);

% =========================================================================

end
