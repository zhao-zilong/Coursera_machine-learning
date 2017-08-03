function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%
for i =1:m
  produit = sigmoid(X(i,:) * theta);
  %fprintf('curent produit: %f\n', log(produit));
  J = J + ( (-y(i,1)) * log(produit) - (1-y(i,1)) * log(1-produit) );
  %fprintf('curent J: %f\n', J);
endfor
J = J/m;

sum = 0;
for p = 1:rows(theta)
  for q = 1:m
    sum += (sigmoid(X(q,:) * theta) - y(q,1))*X(q,p);
  endfor
  grad(p,1) = sum/m;
  sum = 0;
  
endfor

% =============================================================

end
