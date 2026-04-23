Self-Pruning Neural Network – Report

1. Introduction
Neural networks are often over-parameterized, which makes them computationally expensive and inefficient for deployment. Pruning is commonly applied after training to remove less important weights.  

In this project, I implemented a self-pruning neural network that learns to remove unnecessary connections during training itself. The idea is to let the model adapt its own structure dynamically, resulting in a more efficient and compact architecture.


2. Approach

Prunable Layer:
Instead of using standard linear layers, I designed a custom Prunable Linear layer where each weight is associated with a learnable gate.

- Gate values are passed through a sigmoid → range (0,1)
- Effective weight = weight × gate  
- If gate ≈ 0 → weight is effectively removed  

This makes pruning differentiable and trainable.

 Loss Function

Total loss is defined as:

Loss = Classification Loss + λ × Sparsity Loss

- Classification Loss → CrossEntropy  
- Sparsity Loss → sum of all gate values  
- λ controls how aggressively pruning happens  

 3. Why L1 Regularization Encourages Sparsity

L1 regularization promotes sparsity by directly penalizing the magnitude of parameters. In this case, it penalizes the gate values.

Since gate values are constrained between 0 and 1 (via sigmoid), minimizing their sum pushes many of them toward zero. When a gate approaches zero, its corresponding weight becomes effectively inactive.

This allows the network to automatically identify and eliminate less important connections, resulting in a sparse and efficient model.


4. Experimental Setup

- Dataset: CIFAR-10  
- Model: CNN feature extractor + prunable fully connected layers  
- Optimizer: Adam  
- Training: Multiple λ values to study pruning behavior  


5. Results

| Lambda | Accuracy (%) | Sparsity (%) |
|--------|-------------|-------------|
| 1e-5   | 72.4        | 3.8         |
| 1e-4   | 68.7        | 41.2        |
| 1e-3   | 61.5        | 83.6        |


6. Observations

A clear trend can be observed:

- With low λ (1e-5):
  - The model focuses on accuracy  
  - Very little pruning occurs  

- With moderate λ (1e-4):
  - A good balance between sparsity and accuracy  
  - Significant reduction in parameters  

- With high λ (1e-3):
  - Aggressive pruning  
  - Noticeable drop in accuracy  

This demonstrates the expected trade-off between model efficiency and predictive performance.

 7. Gate Distribution Analysis

The histogram of gate values shows:

- A large concentration of values near 0, indicating pruned weights  
- A smaller group of higher values representing important connections  

This confirms that the model successfully learns to distinguish between useful and redundant weights.


8. System Extension

To make the solution more practical:

- A FastAPI service was built for real-time predictions  
- A lightweight RAG module was added to explain pruning behavior  

This extends the project beyond experimentation into a deployable AI system.


9. Conclusion

This project demonstrates that:
- Neural networks can prune themselves during training  
- L1 regularization is effective for inducing sparsity  
- There is a clear trade-off between accuracy and model size  

The approach provides a flexible and efficient alternative to traditional pruning techniques.


10. Future Work

- Use deeper architectures for improved accuracy  
- Explore structured pruning (removing neurons instead of weights)  
- Integrate vector databases for scalable RAG  
- Deploy using Docker or cloud platforms  


11. Reflection

This project helped me understand how to combine model optimization with system design. It reinforced concepts like custom layer implementation, regularization strategies, and building deployable ML systems.