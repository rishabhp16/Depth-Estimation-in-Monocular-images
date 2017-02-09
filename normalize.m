% Normalize matrix A based on matrix B's max and min value
% where A is testing/training data and B is training data
function res = normalize(A,B)
    mx = max(B(:));
    mi = min(B(:));
    res = (A-mi)./(mx - mi);
end