#  FedPSO-MLP: Federated Learning MLP using Particle Swarm Optimization
In this work, we propose a hybrid protocol called FedPSO-MLP that combines particle swarm optimization algorithms with multilayer perceptron models to update the global model by collecting weights from learned models and to provide optimal solutions.<br><br>
The experiments demonstrated that our proposed FedPSO- MLP outperformed the most famous Federated Averaging algorithm FedAvg, achieving an accuracy of 96%. Moreover, it showed an improvement in decreasing error by approximately 75%. <br><br>

<img src='./doc/imgs/Framework Architecture- showing the weight update process of FedPSO-MLP.png' title='Schematic diagram of the proposed FedPSO-MLP
model' >
<center>Schematic diagram of the proposed FedPSO-MLP model</center>

# Requirements
Install all the packages from requirments.txt
<ul>
<li>NumPy
<li>Pandas
<li>scikit-learn
<li>MLPRegressor
</ul>
You will also need to have software installed to run and execute a Pycharm IDE.

# Data

These data were collected and disseminated according to this publication: https://www.nature.com/articles/s41597-020-00582-3

# Running the experiments
The baseline experiment trains the model in the conventional way.

To run the baseline experiment with MNIST on MLP using CPU:<br>
<pre><b> &nbsp; python ./FedPSO_MLP.py </b> </pre>

# Citation
If you find our work useful in your research, please cite:

Aline Abbouda, Mohamed-el-Amine Brahmiab, Rocks Mazraania, Abdelhafid Abouaissac and Ahmad Shahin, <b>"A Hybrid Aggregation Approach for Federated Learning to Improve Energy Consumption in Smart Buildings"</b>, 2022.
