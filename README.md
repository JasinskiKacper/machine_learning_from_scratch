# 🤖 Machine Learning from Scratch

Machine learning models implemented from scratch in Python using only `numpy` and `pandas`.  
Built only for learning purposes to deeply understand the math and mechanics behind each algorithm.

## 📚 Implemented Models

### 1. Linear Regression
- **Algorithms:** Closed form, Gradient descent, Stochastic gradient descent, Mini-batch gradient descent, Singular value decomposition (SVD), Ridge, Lasso, Polynomial regression
- **Dataset:** World Happiness Report
- **Notes:** Implemented multiple optimization methods (closed-form, GD variants). Observed that gradient descent is sensitive to learning rate, while SVD provides a stable and exact solution. Ridge and Lasso helped reduce overfitting on noisy data.

### 2. Logistic Regression
- **Algorithms:** Gradient Descent, Stochastic gradent descent, Mini-Batch gradient descent
- **Dataset:** Iris
- **Notes:** Focused on understanding decision boundaries and gradient-based optimization. Stochastic and mini-batch versions significantly speed up training with slight noise in convergence.

### 3. K-Nearest Neighbors
- **Algorithms:** KNN classifier
- **Dataset:** World Happiness Report
- **Notes:** Simple but effective lazy learning algorithm. Performance strongly depends on feature scaling and choice of k. No training phase makes it fast to implement but slow at prediction time.

### 4. Decision Tree
- **Algorithms:** CART (Gini impurity)
- **Dataset:** Iris
- **Notes:** Key challenge was implementing optimal splits using Gini impurity. Trees easily overfit without depth control. Categorical and numerical splitting required separate handling.

### 5. Naive Bayes
- **Algorithms:** Multinomial naive bayes, Gaussian naive bayes
- **Dataset:** SMS Spam Collection Dataset, Iris
- **Notes:** Multinomial NB works well for text classification due to word frequency modeling. Gaussian NB assumes normal distribution, which performs well on continuous Iris features but is sensitive to variance assumptions.

### 6. Perceptron
- **Algorithms:** Rosenblatt's Perceptron
- **Dataset:** Iris
- **Notes:** Demonstrates linear separability concept. Converges only if data is linearly separable. Very sensitive to feature scaling and learning rate.

### 7. Support Vector Machine
- **Algorithms:** Hard Margin SGD, Soft Margin GD/SGD/Mini-Batch, Kernel SVM (RBF)
- **Dataset:** Iris
- **Notes:** Hard margin works only for perfectly separable data, while soft margin introduces regularization (C parameter). Kernel SVM (RBF) allows non-linear decision boundaries but is computationally expensive.

### 8. Principal Component Analysis
- **Algorithms:** Eigenvectors, Singular value decomposition (SVD)
- **Dataset:** Iris
- **Notes:** Eigenvector and SVD approaches produce the same projection but SVD is numerically more stable. PCA effectively reduces dimensionality while preserving most variance.

### 9. K-Means Clustering
- **Algorithms:** Lloyd's Algorithm, K-Means++
- **Dataset:** Iris
- **Notes:** Lloyd’s algorithm is sensitive to initialization and may converge to local minima. K-Means++ improves stability by smarter centroid initialization.

---

## 🛠️ Tech Stack

- Python 3.x
- NumPy 2.4.2
- Pandas 3.0.1
- Matplotlib 3.10.8
- Seaborn 0.13.2

---

## 🚀 Getting Started

```bash
git clone https://github.com/JasinskiKacper/machine_learning_from_scratch.git
cd machine_learning_from_scratch
pip install -r requirements.txt
```

---

## 📄 License

MIT