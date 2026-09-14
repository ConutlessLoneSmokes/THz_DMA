function [ loss ] = getSpreadLoss( f, d )
%Calculate the spread loss
%  [loss] = getSpreadLoss(f, d)
%Inputs:
%   f: frequency
%   d: distance
%Outputs:
%   loss: spread loss

loss = 20 * log10(4 * pi * f * d / 3e8);

end
