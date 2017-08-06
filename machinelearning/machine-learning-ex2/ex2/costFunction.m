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



% J = 1/m sum(m, i=1, Cost(htheta(Xi), yi))
% Cost(htheta(x), y) = -log(htheta(x))       if y = 1
% Cost(htheta(x), y) = -log(1 - htheta(x))   if y = 0
% htheta(x) = 1/(1 + e^(-x))
%
% J = (1/m) sum(m, i=1, y(i)*-log(sigmoid(x) + (1-y(i))*-log(1-sigmoid(x)))
% gradient = (sigmoid(X * theta) - y) * X ??

for i = 1:m
  sig = sigmoid(X(i,:) * theta);
  J += (1/m) * sum(y(i) * -log(sig) + (1 - y(i)) * -log(1 - sig));
end

for i = 1:size(theta)
  grad(i) = (1/m) * sum((sigmoid(X * theta) - y) .* X(:, i));
end

% =============================================================

end
