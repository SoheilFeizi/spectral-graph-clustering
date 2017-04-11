% Example script illustrating usage of functions in RDPG-package
% Reference: O'Connor LJ, Medard M and Feizi S. "Clustering over the
% Logistic Random Dot Product Graph" 2015, arXiv preprint.


%% Generate an SBM and recover the communities
N = 1000;
clusters = [{1:200}, {201:600}, {601:1000}];
k=numel(clusters); % no. clusters
d=k-1; %dimension of latent-position space
between_cluster_density = .05;
within_cluster_density = [.15, .1, between_cluster_density];%density for each cluster

% Adjacency matrix
A = make_sbm(N, clusters, within_cluster_density, between_cluster_density); 

figure;
subplot(2,2,1)
imagesc(A);colormap(parula(2))
title('SBM adjacency matrix')

% Clustering with RDPG
[est_latent_positions, cluster_assignments]=lrdpg_fit(A,d,k);

% Plotting
subplot(2,2,2); hold on
colors=[{'red'} {'blue'} {'green'} {'cyan'} {'black'} {'yellow'}];
for kk=1:k
    scatter(est_latent_positions(clusters{kk},1),...
        est_latent_positions(clusters{kk},2), colors{mod(kk-1,numel(colors))+1})
end
title('Estimated latent positions of nodes in each community')
legend('First community','Second community','Third community')

% Scoring
score = jaccard_score(clusters, cluster_assignments);
fprintf('Overlap b/t true and est. communities: %f\n',score)

%% Generate a non-SBM RDPG and recover the latent positions
N=1000;
d=1;%dimension of latent position space
latent_positions=randn(N,d)*sqrt(.4);
mu=3;%larger value of mu -> sparser graph

% A - adjacency matrix. idx - orders nodes by LP for visualization
[A, ~, idx] = make_lrdpg(latent_positions,mu);

subplot(2,2,3)
imagesc(A(idx,idx));colormap(parula(2))
title('RDPG adjacency matrix')

% Inference with RDPG
est_latent_positions = lrdpg_fit(A,d);
subplot(2,2,4);
scatter(latent_positions, est_latent_positions)
title('True vs. estimated latent positions')

% Scoring
score = corr(latent_positions,est_latent_positions)^2;
fprintf('r2 b/t true and est. latent positions: %f\n',score)




