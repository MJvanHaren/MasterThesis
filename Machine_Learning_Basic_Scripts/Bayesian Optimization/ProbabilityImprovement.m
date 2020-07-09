function [prob,muTest] = ProbabilityImprovement(x,xTest,y,hyperParameters)
x = x(:); %xBayesian
xTest = xTest(:); % xSamples
%% so-far
bestYSoFar = max(y);

[yTest,muTest,stdvTest] = GPEstimateNoMean(xTest,x,hyperParameters,y);
normalDist = makedist('Normal','mu',0,'sigma',1);
prob = cdf(normalDist,(muTest - bestYSoFar) ./ (stdvTest+1e-8)');
end

