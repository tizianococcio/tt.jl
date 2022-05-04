using StatsBase

function PCA(X)
    X = copy(X)
    Z = StatsBase.fit(ZScoreTransform, X, dims=1)
    StatsBase.transform!(Z, X)
    M = StatsBase.fit(PCA, X; pratio=1, maxoutdim=10);
    return projection(M)' * (X .- mean(M));
end

function group_labels_by_id(labels)
    all_labels = collect(Set(labels))
    sort!(all_labels)
    mapping = Dict(l=>n for (n,l) in enumerate(all_labels))
    return [mapping[l] for l in labels]
end
