# Comprehensive ML/DL Model Taxonomy from Foundational Books

here is some of the models and their basic description that Planned for implementation in this Repo

## I. SUPERVISED LEARNING

### A. REGRESSION (Predict Continuous Values)

1. Linear Regression (Ordinary Least Squares): A fundamental statistical method that models the relationship between a dependent variable and one or more independent variables using a linear equation, minimizing the sum of squared residuals between observed and predicted values to find the best-fitting line.

2. Ridge Regression (L2 regularization): An extension of linear regression that adds L2 regularization to the loss function, penalizing large coefficients to prevent overfitting, particularly useful in multicollinear data by shrinking coefficients towards zero without eliminating them.

3. Lasso Regression (L1 regularization): Similar to ridge regression but uses L1 regularization, which can drive some coefficients to exactly zero, enabling feature selection and producing sparse models that are easier to interpret.

4. Elastic Net Regression (L1 + L2): A hybrid of ridge and lasso regression that combines both L1 and L2 penalties, balancing feature selection with coefficient shrinkage, ideal for datasets with many correlated features.

5. Polynomial Regression: Extends linear regression by including polynomial terms of the independent variables, allowing the model to capture non-linear relationships while still using linear regression techniques on the transformed features.

6. Logistic Regression: Despite its name, a classification algorithm that models the probability of a binary outcome using the logistic sigmoid function, extendable to multinomial cases for multi-class problems, estimating parameters via maximum likelihood.

7. Poisson Regression: A generalized linear model for count data following a Poisson distribution, predicting the expected count by linking the mean to a linear combination of predictors through a log function, handling overdispersion with variants like negative binomial.

8. Bayesian Linear Regression: Incorporates Bayesian inference into linear regression, treating coefficients as random variables with prior distributions, providing probabilistic predictions and uncertainty estimates via posterior distributions.

9. Gaussian Process Regression (Kriging): A non-parametric Bayesian approach that models the target function as a Gaussian process, using kernel functions to define covariance and making predictions with uncertainty estimates, excelling in interpolation tasks with limited data.

### B. CLASSIFICATION (Predict Discrete Labels)

1. Logistic Regression (Binomial & Multinomial): For binomial, predicts binary outcomes; for multinomial, handles multiple classes by extending the logistic function, using softmax for probabilities and maximum likelihood estimation for fitting.

2. Linear Discriminant Analysis (LDA): A linear classification technique that projects data onto a lower-dimensional space to maximize class separability, assuming Gaussian distributions with equal covariances, also used for dimensionality reduction.

3. Quadratic Discriminant Analysis (QDA): Similar to LDA but allows different covariance matrices for each class, enabling quadratic decision boundaries and better handling of classes with varying spreads.

4. Naïve Bayes (Gaussian, Multinomial, Bernoulli): A probabilistic classifier based on Bayes' theorem with independence assumptions; Gaussian for continuous features assuming normal distributions, Multinomial for discrete counts like text, Bernoulli for binary features.

5. k-Nearest Neighbors (k-NN): A non-parametric instance-based algorithm that classifies new points based on the majority vote of the k closest training examples in feature space, using distance metrics like Euclidean, sensitive to choice of k and scaling.

6. Support Vector Machines (SVM) with various kernels:

   - Linear SVM: Finds a hyperplane that maximally separates classes in the original space, using hinge loss and regularization to handle non-separable data.

   - Polynomial Kernel SVM: Maps data to higher dimensions using polynomial functions, enabling non-linear boundaries while computing in kernel space.

   - Radial Basis Function (RBF) Kernel SVM: Uses Gaussian kernels for infinite-dimensional mapping, effective for complex non-linear patterns, with gamma parameter controlling influence radius.

   - Sigmoid Kernel SVM: Employs hyperbolic tangent kernels, similar to neural networks, but less common due to potential non-positive semi-definiteness issues.

7. Decision Trees (CART, ID3, C4.5): Recursive partitioning algorithms; CART handles both regression and classification with Gini or MSE, ID3 uses information gain for categorical data, C4.5 extends ID3 with continuous attributes and pruning.

8. Ensemble Methods:

   - Random Forest (Bagging of decision trees): Builds multiple decision trees on bootstrapped samples and random feature subsets, aggregating predictions via voting or averaging to reduce variance and improve generalization.

   - Gradient Boosted Machines:

     - XGBoost (Extreme Gradient Boosting): An optimized gradient boosting framework with regularization, parallel processing, and handling missing values, using second-order approximations for faster convergence.

     - LightGBM: A gradient boosting method using histogram-based splitting and leaf-wise growth for efficiency on large datasets, supporting categorical features natively.

     - CatBoost: Focuses on categorical data with ordered boosting and symmetric trees, reducing overfitting through random permutations and minimal hyperparameter tuning.

   - AdaBoost (Adaptive Boosting): Sequentially trains weak learners, increasing weights on misclassified examples, combining them with weighted voting for strong classification performance.

   - Stacking (Stacked Generalization): Trains multiple base models and uses a meta-learner to combine their predictions, often via cross-validation to prevent overfitting.

## II. UNSUPERVISED LEARNING

### A. CLUSTERING

1. k-Means Clustering (Lloyd's algorithm): Partitions data into k clusters by minimizing within-cluster variance, iteratively assigning points to nearest centroids and updating centroids until convergence.

2. k-Medoids/PAM (Partitioning Around Medoids): Similar to k-means but uses actual data points as medoids, minimizing dissimilarity to handle outliers better, more robust but computationally intensive.

3. Hierarchical Clustering:

   - Agglomerative (bottom-up): Starts with individual points as clusters, merging closest pairs based on linkage until a single cluster, producing a dendrogram for hierarchy visualization.

   - Divisive (top-down): Begins with all points in one cluster, recursively splitting until individual points, less common due to higher complexity.

   - Linkage methods: Single (nearest neighbor), Complete (farthest), Average (mean distance), Ward (minimizes variance increase).

4. Density-Based Clustering:

   - DBSCAN (Density-Based Spatial Clustering): Groups points in high-density regions separated by low-density areas, identifying core points, border points, and noise, handling arbitrary shapes without specifying cluster count.

   - OPTICS (Ordering Points To Identify Clustering Structure): Extends DBSCAN by producing a reachability plot to extract clusters at varying densities, better for datasets with varying cluster densities.

   - HDBSCAN (Hierarchical DBSCAN): Builds a hierarchy of DBSCAN clusters, selecting stable ones via minimum spanning trees, robust to noise and parameter-sensitive.

5. Distribution-Based Clustering:

   - Gaussian Mixture Models (GMM) with EM algorithm: Assumes data from a mixture of Gaussians, using Expectation-Maximization to estimate parameters, providing soft assignments and handling elliptical clusters.

6. Probabilistic Clustering:

   - Expectation-Maximization (EM) for mixture models: An iterative algorithm alternating expectation (assign probabilities) and maximization (update parameters) steps for latent variable models like GMM.

7. Advanced Clustering:

   - Spectral Clustering: Uses graph Laplacian eigenvalues to embed data, then applies k-means, effective for non-convex clusters by capturing connectivity.

   - Mean Shift Clustering: Iteratively shifts points towards modes of density function, non-parametric and finding arbitrary shaped clusters without preset number.

   - Affinity Propagation: Identifies exemplars via message passing between points based on similarities, automatically determining cluster count.

   - BIRCH (For large datasets): Builds a clustering feature tree for incremental, hierarchical clustering, efficient for streaming data with limited memory.

### B. DIMENSIONALITY REDUCTION

1. Linear Methods:

   - Principal Component Analysis (PCA): Projects data onto orthogonal components that maximize variance, reducing dimensions while retaining most information via eigenvalue decomposition.

   - Factor Analysis: Models observed variables as linear combinations of latent factors plus noise, assuming Gaussian errors, used for identifying underlying structures.

   - Linear Discriminant Analysis (LDA): Supervised projection maximizing class separation, assuming equal covariances, also for classification.

   - Independent Component Analysis (ICA): Separates multivariate signals into additive independent non-Gaussian components, useful for blind source separation like cocktail party problem.

   - Non-negative Matrix Factorization (NMF): Decomposes non-negative matrices into non-negative factors, interpretable for parts-based representations like topic modeling.

   - Singular Value Decomposition (SVD): Factorizes matrices into singular vectors and values, foundational for PCA and compression, handling both dense and sparse data.

2. Manifold Learning (Non-linear):

   - t-SNE (t-Distributed Stochastic Neighbor Embedding): Visualizes high-dimensional data in low dimensions by preserving local similarities via Student's t-distribution, great for clusters but not distances.

   - UMAP (Uniform Manifold Approximation and Projection): Preserves both local and global structure using graph-based approximation, faster and scalable than t-SNE.

   - Isomap: Extends MDS by estimating geodesic distances on manifolds via shortest paths, preserving global geometry for unfolded manifolds.

   - LLE (Locally Linear Embedding): Reconstructs local neighborhoods linearly, then embeds globally, assuming manifold is locally linear.

   - MDS (Multi-Dimensional Scaling): Embeds data in low dimensions preserving pairwise distances, metric or non-metric variants.

   - Autoencoders: Neural networks trained to reconstruct input after compression, learning non-linear embeddings, variants for denoising or sparsity.

### C. ASSOCIATION RULE LEARNING

1. Apriori Algorithm: Mines frequent itemsets using breadth-first search and candidate generation, pruning with support thresholds, then generates rules based on confidence.

2. FP-Growth (Frequent Pattern Growth): Builds a compact FP-tree structure to mine frequent patterns without candidate generation, more efficient for dense datasets.

3. Eclat Algorithm: Uses depth-first search on vertical data format (tid-lists) for frequent itemsets, intersecting lists for efficiency in sparse data.

## III. DEEP LEARNING ARCHITECTURES

### A. FEEDFORWARD NEURAL NETWORKS (FNN)

1. Multilayer Perceptron (MLP): Stacked layers of neurons with non-linear activations, trained via backpropagation, universal approximators for complex functions.

2. Autoencoders:

   - Vanilla Autoencoder: Encodes input to latent space then decodes, minimizing reconstruction error for compression and feature learning.

   - Denoising Autoencoder: Trained on noisy inputs to reconstruct clean ones, improving robustness and feature extraction.

   - Sparse Autoencoder: Adds sparsity penalty to activations, encouraging efficient representations like in biological systems.

   - Contractive Autoencoder: Penalizes sensitivity to input changes via Jacobian, learning robust manifolds.

   - Variational Autoencoder (VAE): Probabilistic encoder-decoder with KL divergence for regularized latent space, enabling generative sampling.

3. Deep Belief Networks (DBN): Stacked Restricted Boltzmann Machines pre-trained layer-wise, then fine-tuned, for deep feature extraction.

4. Restricted Boltzmann Machines (RBM): Bipartite undirected models learning probability distributions over inputs, used for dimensionality reduction and pre-training.

### B. CONVOLUTIONAL NEURAL NETWORKS (CNN/ConvNets)

1. Standard CNN Architectures:

   - LeNet-5: Early architecture for digit recognition with convolutions, pooling, and fully connected layers.

   - AlexNet: Deep CNN with ReLU, dropout, and data augmentation, winner of ImageNet 2012, accelerating deep learning adoption.

   - VGGNet (VGG-16, VGG-19): Uses small 3x3 filters in deep stacks, uniform architecture for strong performance on image classification.

   - Inception (GoogLeNet): Employs inception modules with parallel convolutions of different sizes, efficient with factorized filters.

   - ResNet (Residual Networks): Introduces skip connections to train very deep networks, alleviating vanishing gradients.

   - DenseNet: Connects each layer to all subsequent ones, promoting feature reuse and reducing parameters.

   - Xception: Replaces standard convolutions with depthwise separable ones, inspired by Inception for better efficiency.

   - MobileNet: Uses depthwise separable convolutions and width multipliers for lightweight models on mobile devices.

   - EfficientNet: Scales depth, width, and resolution optimally using compound coefficients for state-of-the-art efficiency.

2. Specialized CNN Components:

   - Convolutional Layers (1D, 2D, 3D): Apply filters to extract local features; 1D for sequences, 2D for images, 3D for volumes.

   - Pooling Layers (Max, Average, Global): Reduce spatial dimensions; max for invariance, average for smoothing, global for fixed-size outputs.

   - Transposed Convolution (Deconvolution): Upsamples feature maps, used in segmentation and generation for reversing pooling.

   - Dilated (Atrous) Convolution: Increases receptive field with gaps in filters, capturing multi-scale context without losing resolution.

   - Depthwise Separable Convolution: Separates spatial and channel convolutions, reducing computations while maintaining performance.

   - U-Net (for segmentation): Encoder-decoder with skip connections, preserving spatial info for precise biomedical image segmentation.

### C. RECURRENT NEURAL NETWORKS (RNN)

1. Vanilla RNN: Processes sequences with recurrent connections, maintaining hidden states, but suffers from vanishing/exploding gradients.

2. Long Short-Term Memory (LSTM): Uses gates (input, forget, output) to control information flow, mitigating gradient issues for long sequences.

3. Gated Recurrent Unit (GRU): Simplified LSTM with update and reset gates, fewer parameters, similar performance for sequences.

4. Bidirectional RNN/LSTM/GRU: Processes sequences in both directions, capturing past and future context for better predictions.

5. Encoder-Decoder Architectures (Seq2Seq): Encodes input sequence to fixed vector, decodes to output, foundational for translation.

### D. ATTENTION & TRANSFORMER ARCHITECTURES

1. Attention Mechanism (Bahdanau, Luong): Weights input elements dynamically; Bahdanau additive for RNNs, Luong dot-product or general.

2. Transformer (Vaswani et al.): Relies on self-attention and feedforward layers, parallelizable with positional encodings for sequences.

3. BERT (Bidirectional Encoder Representations): Pre-trained transformer encoder for bidirectional context, fine-tuned for NLP tasks.

4. GPT (Generative Pre-trained Transformer): Autoregressive decoder-only transformer, pre-trained on vast text for generation and completion.

5. Vision Transformer (ViT): Applies transformers to image patches with positional embeddings, rivaling CNNs on large datasets.

6. Swin Transformer: Hierarchical vision transformer with shifted windows for local attention, efficient for dense predictions.

### E. GENERATIVE MODELS

1. Generative Adversarial Networks (GAN):

   - Vanilla GAN: Generator creates fakes, discriminator distinguishes real/fake, trained adversarially to improve realism.

   - DCGAN (Deep Convolutional GAN): Uses convolutions for stable image generation, with batch norm and LeakyReLU.

   - WGAN (Wasserstein GAN): Uses Wasserstein distance for stable training, enforcing Lipschitz via weight clipping or gradient penalty.

   - CycleGAN: Learns unpaired image-to-image translation via cycle consistency loss, for domain adaptation.

   - StyleGAN: Advanced for high-res images with style-based generator, adaptive instance norm for controllable synthesis.

2. Diffusion Models:

   - DDPM (Denoising Diffusion Probabilistic Models): Adds noise progressively, learns to denoise for generation, step-by-step reversal.

   - Stable Diffusion: Latent diffusion model for text-to-image, efficient in compressed space with conditioning.

3. Normalizing Flows: Invertible transformations mapping data to simple distributions, enabling exact likelihood and sampling.

4. Energy-Based Models: Define energy functions for probability, learning via contrastive divergence or score matching for generation.

### F. GRAPH NEURAL NETWORKS (GNN)

1. Graph Convolutional Networks (GCN): Aggregates neighbor features via spectral or spatial convolutions, for node classification.

2. Graph Attention Networks (GAT): Uses attention to weight neighbors dynamically, multi-head for stability.

3. GraphSAGE: Samples and aggregates neighbors inductively, scalable for large graphs with new nodes.

4. Message Passing Neural Networks (MPNN): General framework where nodes exchange messages, update states iteratively.

### G. DEEP REINFORCEMENT LEARNING (DRL)

1. Deep Q-Network (DQN): Approximates Q-values with CNNs, uses experience replay and target networks for stable off-policy learning.

2. Policy Gradient Methods:

   - REINFORCE: Directly optimizes policy parameters using Monte Carlo gradients, high variance but simple.

   - Actor-Critic: Combines policy (actor) and value (critic) estimation, reducing variance with baseline.

   - A2C/A3C (Advantage Actor-Critic): Synchronous/asynchronous variants, advantage for better gradients, parallel training.

   - PPO (Proximal Policy Optimization): Clips surrogate objective for stable updates, trust region-like.

   - TRPO (Trust Region Policy Optimization): Constrains KL divergence for monotonic improvements, complex optimization.

   - SAC (Soft Actor-Critic): Entropy-regularized for exploration, off-policy with continuous actions.

3. Model-Based RL: Learns environment dynamics model, plans or augments data for efficiency.

4. Inverse Reinforcement Learning (IRL): Infers reward function from expert demonstrations, then learns policy.

## IV. PROBABILISTIC & GRAPHICAL MODELS

### A. BAYESIAN MODELS

1. Naïve Bayes Classifiers: Fast probabilistic classifiers assuming feature independence, variants for different distributions.

2. Bayesian Networks (Directed Graphical Models): DAGs representing conditional dependencies, for inference and learning.

3. Hidden Markov Models (HMM): Models sequences with hidden states and observations, using Viterbi for decoding, Baum-Welch for training.

4. Markov Random Fields (MRF) (Undirected Graphical Models): Undirected graphs for joint distributions, used in image processing.

5. Conditional Random Fields (CRF): Discriminative MRFs for structured prediction, like sequence labeling.

6. Gaussian Processes: Non-parametric priors over functions, kernel-based for regression and classification.

7. Latent Dirichlet Allocation (LDA): Generative model for topic discovery in documents, using Dirichlet priors.

8. Probabilistic Matrix Factorization: Bayesian factorization for recommendations, handling uncertainty.

### B. TEMPORAL MODELS

1. Hidden Markov Models (HMM): As above, for time-series with latent states.

2. Kalman Filter (Linear Dynamic Systems): Recursive Bayesian estimation for linear Gaussian systems, predicting and updating states.

3. Extended Kalman Filter (EKF): Non-linear extension using Jacobian linearization for approximate filtering.

4. Particle Filter (Sequential Monte Carlo): Non-parametric approximation using particles for non-linear/non-Gaussian filtering.

## V. KERNEL METHODS

1. Kernel Ridge Regression: Regularized regression in kernel-induced feature space, for non-linear relationships.

2. Support Vector Machines (with kernels): As above, kernels for non-linear mapping.

3. Kernel PCA: Non-linear dimensionality reduction via kernel trick on PCA.

4. Gaussian Processes (kernel-based): As above, kernels define covariance.

## VI. OPTIMIZATION & TRAINING ALGORITHMS

1. Gradient Descent (Batch, Mini-batch, Stochastic): Iterative optimization minimizing loss; batch full dataset, mini-batch subsets, stochastic single examples for speed.

2. Optimizers:

   - Momentum: Accelerates GD by accumulating velocity, overcoming local minima.

   - Nesterov Accelerated Gradient: Looks ahead in momentum direction for better updates.

   - AdaGrad: Adapts learning rates per parameter based on historical gradients, for sparse data.

   - RMSProp: Divides by root mean square of recent gradients, preventing AdaGrad decay.

   - Adam (Adaptive Moment Estimation): Combines momentum and RMSProp, with bias correction.

   - AdamW: Decouples weight decay from Adam for better regularization.

   - LAMB: Layer-wise adaptive rates for large batches, efficient for big models.

3. Learning Rate Schedulers: Adjust rates over time, like step decay, exponential, or cosine annealing for convergence.

4. Regularization Techniques:

   - L1/L2 Regularization: Penalize weights to prevent overfitting, L1 for sparsity.

   - Dropout: Randomly drops units during training, ensemble effect.

   - Batch Normalization: Normalizes activations per batch, accelerating training.

   - Layer Normalization: Normalizes across features, for RNNs and transformers.

   - Weight Normalization: Decouples weight magnitude and direction.

   - Early Stopping: Halts training when validation performance degrades.

   - Data Augmentation: Generates variations like rotations, increasing dataset diversity.

## Which Models Are Covered in Which Book?

1. Géron's "Hands-On ML" (3rd Ed.)

   - All basic supervised models: Linear/Logistic Regression, SVM, Decision Trees, Random Forest, Gradient Boosting

   - All basic unsupervised: k-Means, DBSCAN, PCA, t-SNE

   - Deep Learning: MLP, CNN, RNN/LSTM, Autoencoders, GANs, Transformers (brief), RL basics

2. Bishop's "Pattern Recognition"

   - Comprehensive probabilistic treatment: Linear Regression/Classification, Neural Networks (pre-deep learning)

   - Kernel Methods: SVM, Gaussian Processes

   - Graphical Models: HMM, MRF, Bayesian Networks

   - Mixture Models: GMM with EM

   - Dimensionality Reduction: PCA, Probabilistic PCA, LDA

   - Ensemble Methods: AdaBoost

3. Goodfellow's "Deep Learning"

   - Deep NN Foundations: MLP, optimization, regularization

   - CNN Architectures: comprehensive theory

   - RNN Architectures: LSTM, GRU, Seq2Seq

   - Generative Models: Autoencoders, GANs, Boltzmann Machines

   - Probabilistic DL: Variational Inference, Monte Carlo methods

   - Attention mechanisms (pre-Transformer)

4. Deisenroth's "Mathematics for ML"

   - Mathematical foundation for: Linear Regression, PCA, GMM, SVM

   - Not a model catalog - focuses on the math behind them

## Progressive Learning Path for Models

**Beginner Phase (Months 1-6):**

1. Linear/Logistic Regression → 2. Decision Trees → 3. Random Forest → 4. k-Means → 5. PCA → 6. MLP

**Intermediate Phase (Months 6-12):**

1. SVM (with kernels) → 2. Gradient Boosting → 3. GMM → 4. Autoencoders → 5. CNN → 6. LSTM

**Advanced Phase (Year 2):**

1. Bayesian Methods → 2. Graphical Models → 3. GANs → 4. Transformers → 5. GNNs → 6. Advanced RL

**Expert/Research Phase (Year 3+):**

Diffusion Models, Causal Models, Meta-Learning, Few-shot Learning, Neural ODEs, etc.

## Critical Insight for Beginners:

Don't be overwhelmed by this list. Mastery follows a power law:

- 20% of models (Linear Regression, Logistic Regression, Random Forest, CNN, Transformer) solve 80% of practical problems

- The deepest value comes from understanding mathematical foundations (from Bishop and Goodfellow) which let you understand any new model that emerges

- Focus on intuition first (from Géron), then theory (from Bishop/Goodfellow)

- Build one model from each category deeply before exploring variations

This taxonomy represents the complete "model universe" you'll eventually navigate. Start with the basics in Géron, build mathematical maturity with Bishop, then specialize with Goodfellow. After these three books, you'll be prepared to understand any ML/DL model in research papers.
