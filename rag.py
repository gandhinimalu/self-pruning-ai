docs = [
    "L1 regularization forces gate values toward zero creating sparsity",
    "Higher lambda increases pruning but reduces accuracy",
    "Sigmoid gating determines whether weights remain active"
]

def retrieve(query):
    # simple version (can upgrade later)
    return docs[:2]