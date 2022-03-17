function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
% X is 5000 by 400 cols

layer_1 = [ones(m, 1) X]
% theta1 = 25 by 401
% layer_1 = 5000 by 401

layer_2 = layer_1 * Theta1'
layer_2 = sigmoid(layer_2)
% layer_2 = 5000 by 25

disp("Layer 2 Computed")
%pause;

layer_2 = [ones(m,1) layer_2]
% layer_2 = 5000 by 26

% Theta2 = 10 by 26
layer_3 = layer_2 * Theta2'
layer_3 = sigmoid(layer_3)
% layer_3 = 5000 by 10
% layer_3 = h(x) = hypothesis

disp("All Layers Computed")
%pause;

% getY_inVector gives a vector of 5000 by 10
y_vec = getY_inVector(y, num_labels) 

disp("getY_inVector completed")
%pause;

% inner_K_value :
% the inner summation that goes from k = 1 to k = num_labels
% inner_K_value = 5000 by 10

inner_K_value = -(y_vec .* log(layer_3)) - ((1 - y_vec) .* log(1 - layer_3))

disp("Inner K summation value computed")
%pause;

% this sum below is the second summation in the cost function formula
J = (1 / m) * sum(inner_K_value(:))

disp("Cost J computed")
%pause;

% Regularization for Cost Function
cost = 0
temp_1 = Theta1(:, 2:end)
temp_1 = temp_1(:) .^ 2
cost = cost + sum(temp_1)

temp_2 = Theta2(:, 2:end)
temp_2 = temp_2(:) .^ 2
cost = cost + sum(temp_2)

cost = (lambda / (2 * m)) * cost

J = J + cost


% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%

% 
Triangle_delta_1 = zeros(size(Theta1));

Triangle_delta_2 = zeros(size(Theta2));


for t=1: m
    a_1 = X(t, :)
    % a_1 = 1 by 400
    a_1 = [1 a_1]
    % a_1 = 1 by 401

    % size(a_1, 2) = 401
    % thus W = 25 by 401
    % Not Useful W_1 = randInitializeWeights(size(a_1, 2)-1, 25) %since bias already added above, the function randInitializeWeights also considers the bias term

    % Theta1 = 25 by 401
    z_2 = a_1 * Theta1'
    % z_2 = 1 by 25
    a_2 = sigmoid(z_2)
    % a_2 = 1 by 25
    a_2 = [1 a_2]
    % a_2 = 1 by 26

    % Not Useful W_2 = randInitializeWeights(size(a_2, 2)-1, 10)
    % thus W = 10 by 26

    % Theta2 = 10 by 26
    z_3 = a_2 * Theta2'
    % z_3 = 1 by 10
    a_3 = sigmoid(z_3)
    % a_3 = 1 by 10

    y_vec = zeros(1, num_labels) 
    y_vec(1, y(t, 1)) = 1
    % y_vec = 1 by 10


    delta_3 = a_3 - y_vec
    % delta_3 = 1 by 10


    % Theta2 = 10 by 26
    % z_2 = 1 by 25
    % adding a0=1 in z_2
    delta_2 =  (delta_3 * Theta2) .* sigmoidGradient([1 z_2])
    % delta_2 = 1 by 26


    % remove delta(0)
    delta_2 = delta_2(:, 2:end)
    % delta_2 = 1 by 25

%###issue in regularization
    %solved
    % i was doing regularization per iteration in for loop, though correctly but not in correct place
    % Regularization is to be done once all the derivatives are calculated for all training data
    % Regularization needs theta values which don't change untill the program has once gone through all data points in training set
    % So, in conclusion, regularization once, using the theta matrices, after the for loop, in the end just before returning the capital D values.



    % Theta2 = 10 by 26
    %% regularization_term_2 = Theta2(:, 2:end)
    % excluding the first bias column from Theta2
    % regularization_term_2 = 10 by 25




    %% regularization_term_2 = [zeros(size(regularization_term_2, 1), 1) regularization_term_2]
    % adding a column of bias which was removed so dimension goes back to : 10 by 26


    % delta_3 = 1 by 10
    % a_2 = 1 by 26
    % Triangle_delta_2 = 10 by 26  
    Triangle_delta_2 = Triangle_delta_2 .+ (delta_3' * a_2) %%.+ (lambda * regularization_term_2)
    % delta_3' * a_2 = 10 by 26
    % including a0 in a_2 

    %% regularization_term_1 = Theta1(:, 2:end)
    % Theta1 = 25 by 401
    % excluding the first bias column from Theta1
    % regularization_term_2 = 25 by 400

    %% regularization_term_1 = [zeros(size(regularization_term_1, 1), 1) regularization_term_1]
    % adding a column of zeros so dimension goes back to : 25 by 401

    % delta_2 = 1 by 25 .. after removing delta0 from delta_2
    % a_1 = 1 by 401
    % Triangle_delta_1 = 25 by 401
    Triangle_delta_1 = Triangle_delta_1 .+ (delta_2' * a_1) %%.+ (lambda * regularization_term_1)
    % delta_2' * a_1 = 25 by 401
    % including a0 in a_1 



    





end
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%




% according to me these are basically the finally capital D values in the course week 5

% regularization
reg_term_1 = lambda * Theta1(:, 2:end)
reg_term_1 = [zeros(size(reg_term_1, 1), 1) reg_term_1]

reg_term_2 = lambda * Theta2(:, 2:end)
reg_term_2 = [zeros(size(reg_term_2, 1), 1) reg_term_2]

Triangle_delta_1 = Triangle_delta_1 + reg_term_1

Triangle_delta_2 = Triangle_delta_2 + reg_term_2



Theta1_grad = (1/m) * Triangle_delta_1

Theta2_grad = (1/m) * Triangle_delta_2



% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
