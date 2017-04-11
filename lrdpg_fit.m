function [ v,cell,clustvec ] = lrdpg_fit( A,d,k )
%LRDPG_FIT fits the logistic random dot product graph model to a graph with
%adjacency matrix A.
%   Inputs: A, the symmetric adjacency matrix. Diagonal elements will be
%   ignored. d, the number of dimensions of the latent-position space. k,
%   the number of clusters (if not given, then clustering is not performed)
%   Reference: O'Connor LJ, Medard M and Feizi S. "Clustering over the
%   Logistic Random Dot Product Graph" 2015, arXiv preprint.

n=length(A);
if any(vectorize(A)~=vectorize(A'))
    error('Input a symmetric matrix')
end
pbar=mean(vectorize(A));
mu=-log(pbar./(1-pbar));

% Get eigenvectors of B
[v,~]=eigs(A-pbar,d,'la');

% Construct inputs to regression
Xv=zeros(n*(n-1)/2,d);
for dd=1:d
    Xv(:,dd)=vectorize(v(:,dd)*v(:,dd)');
end
y=[vectorize(A),ones(n*(n-1)/2,1)];

% Run logistic regression of y on Xv, with intercept -mu
lambda=custom_regression(Xv,y,-mu);
lambda=lambda(2:end);%first entry of lambda is the intercept

% Scale latent positions by inferred intercepts
v=v*diag(sqrt(lambda));

if exist('k')
    % k means clustering
    clustvec=kmeans(v,k,'emptyaction','drop','replicates',10);
    
    % convert vector into cell array
    for k=1:max(clustvec)
        cell{k}=find(clustvec==k);
    end
else
    clustvec=[]; cell=[];
end

    function vec= vectorize(mat)
        temp=triu(0 - ones(n));
        temp=temp+ones(n)==1;
        vec=mat(temp(:));
        
    end

end

