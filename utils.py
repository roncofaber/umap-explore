def make_key(n_neighbors, min_dist, n_components, metric, scale):
    return f"{n_neighbors}_{min_dist}_{n_components}_{metric}_{scale}"
