function y_vec = getY_inVector(y, num_labels)
	% here input y is a vecotor of dimension m by 1
	% y_vec output is a vector of dimension m by num_labels
	m = size(y, 1);

	y_vec = zeros(m, num_labels);
	
	for iter=1: m
		y_vec(iter, y(iter, 1)) = 1;
	end

	% y_vec would be 5000 by 10 
	% where each row gives numeric value in vector form/
	% i.e. for 5 === [0 0 0 0 1 0 0 0 0 0]
	% i.e. for 10 === [0 0 0 0 0 0 0 0 0 1]
end