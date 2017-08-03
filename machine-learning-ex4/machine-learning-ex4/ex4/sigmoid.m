function g = sigmoid(z)
%SIGMOID Compute sigmoid functoon
%   J = SIGMOID(z) computes the sigmoid of z.

%g = 1.0 ./ (1.0 + exp(-z));

row = rows(z);
col = columns(z);

for i = 1:row
  for j = 1:col
    g(i,j) = 1.0 ./(1.0 +exp(-z(i,j)))
  endfor

endfor
end
