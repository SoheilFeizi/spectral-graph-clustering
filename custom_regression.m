function [bb] = custom_regression(x,y,intercept)
%CUSTOM_REGRESSION is based on MATLAB function glmfit(). The intercept
%value is fixed, and the coefficients are constrained to be nonnegative. 
%   Input: predictor variable x, response y with two columns (# successes,
%   # trials), fixed intercept value
%   intercept.
%   Output: bb, nonnegative logistic regression coefficients. The first
%   coefficient is the intercept.

distr = 'binomial';

% Set distribution-specific defaults.
N = []; % needed only for binomial

if size(y,2) == 1
    % N will get set to 1 below
    if any(y < 0 | y > 1)
        error(message('stats:glmfit:BadDataBinomialFormat'));
    end
elseif size(y,2) == 2
    y(y(:,2)==0,2) = NaN;
    N = y(:,2);
    y = y(:,1) ./ N;
    if any(y < 0 | y > 1)
        error(message('stats:glmfit:BadDataBinomialRange'));
    end
else
    error(message('stats:glmfit:MatrixOrBernoulliRequired'));
end
% Wait until N has NaNs removed to define variance function p*(1-p)/N and
% the deviance function 2*(y*log(y/mu) + (N-y)*log((N-y)/(N-mu))).


% Pad x with ones
x = [ones(size(x,1),1) x];
    
dataClass = superiorfloat(x,y);
x = cast(x,dataClass);
y = cast(y,dataClass);

% Link function, its derivative and inverse
linkFun=@(mu)log(mu./(1-mu));
dlinkFun=@(mu)1./(mu.*(1-mu));
ilinkFun=@(eta)1./(1+exp(-eta));

% If x is rank deficient (perhaps because it is overparameterized), we will
% warn and remove columns, and the corresponding coefficients and std. errs.
% will be forced to zero.
[n,ncolx] = size(x);

% Number of observations after removing missing data, number of coeffs after
% removing dependent cols and (possibly) adding a constant term.
[n,p] = size(x);

% Define variance and deviance for binomial, now that N has NaNs removed.
if isequal(distr, 'binomial')
    sqrtN = sqrt(N);
    sqrtvarFun = @(mu) sqrt(mu).*sqrt(1-mu) ./ sqrtN;
    devFun = @(mu,y) 2*N.*(y.*log((y+(y==0))./mu) + (1-y).*log((1-y+(y==1))./(1-mu)));
end

% Initialize mu and eta from y.
mu = startingVals(distr,y,N);
eta = linkFun(mu);
pwts = ones(length(y),1);

% Set up for iterations
iter = 0;
iterLim = 100;
warned = false;
seps = sqrt(eps);
convcrit = 1e-6;
b = zeros(p,1,dataClass);
offset=0;

% Enforce limits on mu to guard against an inverse link that doesn't map into
% the support of the distribution.

% mu is a probability, so order one is the natural scale, and eps is a
% reasonable lower limit on that scale (plus it's symmetric).
muLims = [eps(dataClass) 1-eps(dataClass)];


while iter <= iterLim
    iter = iter+1;
    
    % Compute adjusted dependent variable for least squares fit
    deta = dlinkFun(mu);
    z = eta + (y - mu) .* deta;
    
    % Compute IRLS weights the inverse of the variance function
    sqrtirls = abs(deta) .* sqrtvarFun(mu);
    sqrtw = sqrt(pwts) ./ sqrtirls;
    
    % If the weights have an enormous range, we won't be able to do IRLS very
    % well.  The prior weights may be bad, or the fitted mu's may have too
    % wide a range, which is probably because the data do as well, or because
    % the link function is trying to go outside the distribution's support.
    wtol = max(sqrtw)*eps(dataClass)^(2/3);
    t = (sqrtw < wtol);
    if any(t)
        t = t & (sqrtw ~= 0);
        if any(t)
            sqrtw(t) = wtol;
            if ~warned
                warning(message('stats:glmfit:BadScaling'));
            end
            warned = true;
        end
    end
    
    % Compute coefficient estimates for this iteration - the IRLS step
    b_old = b;
    [b,R] = wfit(z - offset, x, sqrtw);
    b(1)=intercept;
    b(2:end)=max(0,b(2:end));
    
    % Form current linear predictor, including offset
    eta = offset + x * b;
    
    % Compute predicted mean using inverse link function
    mu = ilinkFun(eta);
    
    % Force mean in bounds, in case the link function is a wacky choice
    
    if any(mu < muLims(1) | muLims(2) < mu)
        mu = max(min(mu,muLims(2)),muLims(1));
    end
    
    
    % Check stopping conditions
    if (~any(abs(b-b_old) > convcrit * max(seps, abs(b_old)))), break; end
end
if iter > iterLim
    warning(message('stats:glmfit:IterationLimit'));
end

bb = b;

if iter>iterLim && isequal(distr,'binomial')
    diagnoseSeparation(eta,y,N);
end

if nargout > 1
    % Sum components of deviance to get the total deviance.
    di = devFun(mu,y);
    dev = sum(pwts .* di);
end




function [b,R] = wfit(y,x,sw)
% Perform a weighted least squares fit
[~,p] = size(x);
yw = y .* sw;
xw = x .* sw(:,ones(1,p));
% No pivoting, no basic solution.  We've removed dependent cols from x, and
% checked the weights, so xw should be full rank.
[Q,R] = qr(xw,0);
b = R \ (Q'*yw);


function mu = startingVals(distr,y,N)
% Find a starting value for the mean, avoiding boundary values
switch distr
    case 'poisson'
        mu = y + 0.25;
    case 'binomial'
        mu = (N .* y + 0.5) ./ (N + 1);
    case {'gamma' 'inverse gaussian'}
        mu = max(y, eps(class(y))); % somewhat arbitrary
    otherwise
        mu = y;
end


function diagnoseSeparation(eta,y,N)
% Compute sample proportions, sorted by increasing fitted value
[x,idx] = sort(eta);
if ~isscalar(N)
    N = N(idx);
end
p = y(idx);
if all(p==p(1))   % all sample proportions are the same
    return
end
if x(1)==x(end)   % all fitted probabilities are the same
    return
end

noFront = 0<p(1) && p(1)<1;     % no "front" section as defined below
noEnd = 0<p(end) && p(end)<1;   % no "end" section as defined below
if p(1)==p(end) || (noFront && noEnd)
    % No potential for perfect separation if the ends match or neither
    % end is perfect
    return
end

% There is at least one observation potentially taking probability 0 or
% 1 at one end or the other with the data sorted by eta. We want to see
% if the data, sorted by eta (x) value, have this form:
%        x(1)<=...<=x(A)  <  x(A+1)=...=x(B-1)  <  x(B)<=...<=x(n)
% with   p(1)=...=p(A)=0                           p(B)=...=p(n)=1
% or     p(1)=...=p(A)=1                           p(B)=...=p(n)=0
%
% This includes the possibilities:
%     A+1=B  - no middle section
%     A=0    - no perfect fit at the front
%     B=n+1  - no perfect fit at the end
dx = 100*max(eps(x(1)),eps(x(end)));
n = length(p);
if noFront
    A = 0;
else
    A = find(p~=p(1),1,'first')-1;
    cutoff = x(A+1)-dx;
    A = sum(x(1:A)<cutoff);
end

if noEnd
    B = n+1;
else
    B = find(p~=p(end),1,'last')+1;
    cutoff = x(B-1)+dx;
    B = (n+1) - sum(x(B:end)>cutoff);
end

if A+1<B-1
    % There is a middle region with >1 point, see if x varies there
    if x(B-1)-x(A+1)>dx
        return
    end
end

% We have perfect separation that can be defined by some middle point
if A+1==B
    xmid = x(A) + 0.5*(x(B)-x(A));
else
    xmid = x(A+1);
    if isscalar(N)
        pmid = mean(p(A+1:B-1));
    else
        pmid = sum(p(A+1:B-1).*N(A+1:B-1)) / sum(N(A+1:B-1));
    end
end

% Create explanation part for the lower region, if any
if A>=1
    explanation = sprintf('\n   XB<%g: P=%g',xmid,p(1));
else
    explanation = '';
end

% Add explanation part for the middle region, if any
if A+1<B
    explanation = sprintf('%s\n   XB=%g: P=%g',explanation,xmid,pmid);
end

% Add explanation part for the upper region, if any
if B<=n
    explanation = sprintf('%s\n   XB>%g: P=%g',explanation,xmid,p(end));
end

warning(message('stats:glmfit:PerfectSeparation', explanation));


