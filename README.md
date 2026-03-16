# Mini AI Model Trainer Framework 🤖

A lightweight Python framework that simulates how machine learning models are configured, trained, and evaluated. This project demonstrates core Object-Oriented Programming (OOP) design patterns used in professional libraries like PyTorch and Scikit-Learn.

## 🛠️ OOP Concepts Implemented

This framework serves as a practical demonstration of the following concepts:

| Concept | Application in Code |
| :--- | :--- |
| **Abstraction** | `BaseModel` uses `abc.ABC` to ensure every model has a `train` and `evaluate` method. |
| **Inheritance** | `LinearRegressionModel` and `NeuralNetworkModel` inherit from `BaseModel`. |
| **Polymorphism** | The `Trainer` class runs any model type through the same `.run()` interface. |
| **Composition** | `BaseModel` contains a `ModelConfig` instance to manage settings. |
| **Aggregation** | `Trainer` receives a `DataLoader` externally, allowing the data to exist independently. |
| **Class Attributes** | `BaseModel.model_count` tracks the total number of models initialized. |
| **Super()** | Child classes use `super().__init__()` to maintain the logic of the parent class. |

## 🚀 Quick Start

### 1. Prerequisites
* Python 3.x installed on your system.

### 2. Installation
Clone this repository or download the files:
```bash
git clone https://github.com/YOUR_USERNAME/MiniAITrainer.git
cd MiniAITrainer