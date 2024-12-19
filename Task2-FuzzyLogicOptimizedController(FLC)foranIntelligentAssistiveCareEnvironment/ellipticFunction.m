function f = ellipticFunction(x)
    % Elliptic Function: Sum of scaled squares
    f = sum((10.^6).^((0:(length(x)-1))/(length(x)-1)) .* x.^2);
end
