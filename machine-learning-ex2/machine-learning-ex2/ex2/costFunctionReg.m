function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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

for i =1:m
  produit = sigmoid(X(i,:) * theta);
  %fprintf('curent produit: %f\n', log(produit));
  J = J + ( (-y(i,1)) * log(produit) - (1-y(i,1)) * log(1-produit) );
  %fprintf('curent J: %f\n', J);
endfor

for i = 2:rows(theta)
  J += lambda/2 * theta(i,1)^2;
endfor

J = J/m;

sum = 0;
for p = 1:rows(theta)
  for q = 1:m
    sum += (sigmoid(X(q,:) * theta) - y(q,1))*X(q,p);
  endfor
  if(p == 1)
    grad(p,1) = sum/m;
  else
    grad(p,1) = sum/m + lambda/m*theta(p,1);
  endif
  sum = 0;
  
endfor




% =============================================================

end
