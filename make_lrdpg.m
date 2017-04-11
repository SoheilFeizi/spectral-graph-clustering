function [ A,v_pc,idx,P ] = make_lrdpg( v,mu )
%MAKE_LRDPG generates a graph from given latent positions v and intercept
%mu
%   Inputs: v, an nxd vector. mu, a scalar. Larger mu -> sparser graph
%   Outputs: A, the adjacency matrix. v_pc, the principle components of the
%   latent position space (for visualization). idx, indices of each node
%   sorted by the first PC (for visualization). P, the matrix of
%   probabilities used to generate A.

n=size(v,1);
X=v*v';
P=1./(1+exp(mu-X));
A=rand(n)<P;
A=triu(A);
A=A+A';
A=A-diag(diag(A));

[u,s]=svd(v);
v_pc=u*sqrt(s);
[~,idx]=sortrows(v_pc);
%A=A(idx,idx);
end

