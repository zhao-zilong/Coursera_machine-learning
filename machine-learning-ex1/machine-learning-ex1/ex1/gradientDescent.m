function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
row = rows(theta);
delta = zeros(row,1);
for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    temp = 0;
    for j = 1:row
      for i = 1:m
        temp += (X(i,:) * theta - y(i,1))*X(i,j);
      endfor
      delta(j,1) = temp;
      temp = 0;
    endfor
    theta = theta - alpha/m*delta;
    
      



    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
    % fprintf('\n current J = %f\n', J_history(iter));

end

end
