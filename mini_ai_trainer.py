from abc import ABC, abstractmethod

# -------------------- Model Configuration --------------------
class ModelConfig:
    def __init__(self, model_name, learning_rate=0.01, epochs=10):
        self.model_name = model_name
        self.learning_rate = learning_rate
        self.epochs = epochs
    
    def __repr__(self):
        return f"[Config] {self.model_name} | lr={self.learning_rate} | epochs={self.epochs}"


# -------------------- Abstract Base Model --------------------
class BaseModel(ABC):
    model_count = 0  # Class attribute to track number of models created
    
    def __init__(self, config: ModelConfig):
        self.config = config  # Composition: BaseModel owns a ModelConfig
        BaseModel.model_count += 1
    
    @abstractmethod
    def train(self, data):
        pass
    
    @abstractmethod
    def evaluate(self, data):
        pass


# -------------------- Linear Regression Model --------------------
class LinearRegressionModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)  # Call parent init
    
    def train(self, data):
        print(f"LinearRegression: Training on {len(data)} samples for {self.config.epochs} epochs (lr={self.config.learning_rate})")
    
    def evaluate(self, data):
        print("LinearRegression: Evaluation MSE = 0.042")


# -------------------- Neural Network Model --------------------
class NeuralNetworkModel(BaseModel):
    def __init__(self, config, layers):
        super().__init__(config)  # Call parent init
        self.layers = layers  # Extra attribute
    
    def train(self, data):
        print(f"NeuralNetwork {self.layers}: Training on {len(data)} samples for {self.config.epochs} epochs (lr={self.config.learning_rate})")
    
    def evaluate(self, data):
        print("NeuralNetwork: Evaluation Accuracy = 91.5%")


# -------------------- Data Loader --------------------
class DataLoader:
    def __init__(self, dataset):
        self.dataset = dataset  # Aggregation: DataLoader exists outside Trainer


# -------------------- Trainer --------------------
class Trainer:
    def __init__(self, model: BaseModel, dataloader: DataLoader):
        self.model = model
        self.dataloader = dataloader
    
    def run(self):
        print(f"--- Training {self.model.config.model_name} ---")
        self.model.train(self.dataloader.dataset)  # Polymorphism: works for any BaseModel
        self.model.evaluate(self.dataloader.dataset)


# -------------------- Example Usage --------------------
if __name__ == "__main__":
    # Dummy dataset
    data = [1, 2, 3, 4, 5]
    
    # Create model configurations
    lr_config = ModelConfig("LinearRegression", learning_rate=0.01, epochs=10)
    nn_config = ModelConfig("NeuralNetwork", learning_rate=0.001, epochs=20)
    
    # Print configs
    print(lr_config)
    print(nn_config)
    
    # Create models
    lr_model = LinearRegressionModel(lr_config)
    nn_model = NeuralNetworkModel(nn_config, layers=[64, 32, 1])
    
    # Print total models created
    print(f"Models created: {BaseModel.model_count}")
    
    # Create DataLoader
    dataloader = DataLoader(data)
    
    # Run training & evaluation
    trainer1 = Trainer(lr_model, dataloader)
    trainer1.run()
    
    trainer2 = Trainer(nn_model, dataloader)
    trainer2.run()