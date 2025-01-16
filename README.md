### Predicting House Prices Using a Feedforward Neural Network

---

### **Table of Contents**
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Requirements](#requirements)
4. [Model Architecture](#model-architecture)
5. [How to Run the Code](#how-to-run-the-code)
6. [Results](#results)
7. [Future Improvements](#future-improvements)
8. [Acknowledgments](#acknowledgments)

---

### **1. Introduction**
This project focuses on developing a predictive model using a feedforward neural network to estimate house prices based on features such as median income, number of rooms, and population in the California Housing dataset. By leveraging PyTorch, the goal is to construct and train a simple yet effective neural network capable of handling regression tasks.

- **Objective**: Build and train a feedforward neural network to predict house prices using the California Housing dataset. The model aims to minimize the error between predicted and actual house prices.
- **Motivation**: This project serves as a practical demonstration of applying deep learning techniques to a real-world regression problem. It highlights the steps involved in preprocessing data, designing a neural network, and evaluating its performance, making it a valuable learning experience for beginners in machine learning and deep learning.

---

### **2. Dataset**
The dataset used in this project is the **California Housing dataset**, available through the `scikit-learn` library.

- **Source**: California Housing dataset from [`scikit-learn`](https://scikit-learn.org/stable/).
- **Features**:
  - **Median Income (`MedInc`)**: Median income of the population in the block.
  - **Average Rooms (`AveRooms`)**: Average number of rooms per household.
  - **Population**: Total population in the block.
  - **Latitude/Longitude**: Geographical location of the block.
- **Target Variable**: Median house value in each block group, represented in hundreds of thousands of dollars.

**Preprocessing Steps**:
1. **Standardization**: The features were scaled to have zero mean and unit variance using `StandardScaler` to ensure the neural network trains effectively.
2. **Train-Test Split**: The dataset was divided into:
   - **Training Set (80%)**: Used to train the model.
   - **Testing Set (20%)**: Used to evaluate the model's performance on unseen data.  

This preprocessing ensures the data is prepared for the neural network, improving training stability and performance.
---

### **3. Requirements**

The following tools and libraries are required to run this project:

```plaintext
- Python 3.8 or higher
- PyTorch: For building and training the neural network.
- scikit-learn: For loading the dataset, preprocessing, and splitting data.
- pandas: For data manipulation and analysis.
- matplotlib: For visualizing results and exploring data.
- seaborn: For advanced data visualization and correlation analysis.
```

To install all dependencies, ensure you have `pip` installed and run the following command:

```bash
pip install -r requirements.txt
```

You can manually install the libraries with the following command:

```bash
pip install torch scikit-learn pandas matplotlib seaborn
``` 

---

### **4. Model Architecture**

The feedforward neural network used in this project is designed to predict house prices based on the features provided in the California Housing dataset. The architecture is structured as follows:

- **Input Layer**:  
  - The input layer size matches the number of features in the dataset (8 features for the California Housing dataset).
  
- **Hidden Layers**:  
  - **1st Layer**: Consists of 64 neurons with ReLU activation for non-linear transformations.  
  - **2nd Layer**: Consists of 32 neurons with ReLU activation to further capture complex patterns in the data.  

- **Output Layer**:  
  - A single neuron with no activation function, suitable for regression tasks, outputs the predicted house price.

- **Loss Function**:  
  - Mean Squared Error (MSE) is used as the loss function, as it is appropriate for measuring prediction accuracy in regression tasks.

- **Optimizer**:  
  - The **Stochastic Gradient Descent (SGD)** optimizer is used to train the model by iteratively adjusting the weights based on the computed gradients.

---

### **5. How to Run the Code**
Step-by-step guide to execute the project:
1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/house-price-prediction
   cd house-price-prediction
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the training script**:
   ```bash
   python train.py
   ```

4. **Evaluate the model**:
   ```bash
   python evaluate.py
   ```

---

### **6. Results**
- **Training Loss**: Decreases over epochs.
- **Test Loss (MSE)**: Final test loss is 0.30

#### Loss Curve
The loss curve shows how the model's loss decreases over training epochs, indicating the model's learning progress.

![Loss Curve](plots/loss_curve.png)

#### Predicted vs. Actual House Prices
The scatterplot compares predicted house prices to actual values. Points closer to the red diagonal line indicate better predictions.

![Predicted vs. Actual House Prices](plots/predicted_vs_actual.png)

---

### **7. Future Improvements**
- Explore Advanced Architectures: Experiment with deeper neural networks or more complex architectures, such as residual networks or additional hidden layers, to better capture non-linear relationships in the data.
- Try Alternative Optimizers: Use optimizers like RMSprop or AdamW to potentially achieve faster convergence and better results compared to SGD.
- Regularization Techniques: Incorporate techniques such as dropout to prevent overfitting or batch normalization to stabilize training and improve generalization.

---

### **8. Acknowledgments**
- California Housing dataset: [scikit-learn](https://scikit-learn.org/stable/).
- PyTorch documentation: [PyTorch](https://pytorch.org/).
