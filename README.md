This project implements and compares several optimization algorithms for federated learning in a simulated distributed environment. The goal is to evaluate and analyze the performance of these algorithms on classification tasks with heterogeneous clients.

The implemented algorithms include:

FedAvg

FedAdam

FedProx

Local SGD

The experiments are designed to study metrics such as global accuracy, communication overhead, robustness, and scalability.

⚙ Installation
Clone the repository:
python -m venv venv
venv\Scripts\activate 
git clone https://github.com/0x0ilyass/Federated_learning.git
cd Federated_learning
pip install --upgrade pip
pip install torch matplotlib numpy torchvision tensorboard tqdm


▶ How to Run
python main.py

Open the example notebook on Google Colab:
https://colab.research.google.com/drive/1nP-C4e8pDOHOPOXBx-GSB-1Gqb5QrPnR?usp=sharing

📊 Datasets
The code automatically downloads and uses the following datasets:
MNIST
CIFAR-10

📈 Metrics
The project reports:
Global accuracy
Communication overhead
Robustness measures
Scalability evaluation
