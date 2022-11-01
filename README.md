#  FedPSO-MLP: Federated Learning MLP using Particle Swarm Optimization
With growth of the world’s population and with challenges of urbanization, it is necessary to develop all infrastructures while reducing greenhouse gas emissions. According to estimates, around one-third of global greenhouse gas emissions come from buildings. Hence, investing in smart buildings can help cut down on greenhouse gas emissions and will certainly have a significant impact on the environment. Through the data collected by smart buildings, machine learning can train a deep
neural network to analyze and predict the energy consumption of their users. Our objective is to develop a platform that will allow users to manage their energy consumption efficiently. this will be done through a combination of optimization techniques and Federated learning. Hence, In this paper, we propose a heuristic
algorithm FedPSO-MLP that uses particle swarm optimization with the help of regressive predictive analysis MLP regressor, one of the Machine Learning (ML) model to update the global model by collecting weights from learned models and to provide the optimal solution. FedPSO-MLP is evolving the way of the data that clients transmit to servers.

# Install
This project requires Python and the following Python libraries installed:

NumPy, 
Pandas, 
scikit-learn, 
MLPRegressor.

You will also need to have software installed to run and execute a Pycharm IDE.

# Data

These data were collected and disseminated according to this publication: https://www.nature.com/articles/s41597-020-00582-3

# Running the experiments
The baseline experiment trains the model in the conventional way.

To run the baseline experiment with MNIST on MLP using CPU:<br>
&nbsp; python ./FedPSO_MLP.py
