function [ jaccard_index ] = jaccard_score( clust,trueclust )
%JACCARD_SCORE computes the normalized jaccard index, a measure of cluster quality, on
%clustered nodes. Small clusters are given equal weight as large ones.

K=numel(trueclust);
while numel(clust)<K
    clust=[clust,{}];
end
if K<6 %For a small # of clusters, try all permutations
    sig=perms(1:K);
    index(1:size(sig,1))=0;
    for ii=1:size(sig,1)
        for jj=1:K
            index(ii)=index(ii)+numel(intersect(clust{jj},trueclust{sig(ii,jj)}))...
                /numel(trueclust{sig(ii,jj)});
        end
    end
    jaccard_index=max(index(:));
else %Otherwise use greedy matching (usually works well)
    remaining(1:K)=true;
    array=1:K;
    jaccard_index=0;
    
    for ii=1:K
        temp(1:K)=0;
        for jj=array(remaining)
            temp(jj)=numel(intersect(clust{ii},trueclust{jj}));
        end
        [overlap,ind]=max(temp);
        remaining(ind)=false;
        jaccard_index=jaccard_index+overlap/numel(trueclust{ind});
    end
end
jaccard_index=(jaccard_index-1)/(numel(trueclust)-1);
end

