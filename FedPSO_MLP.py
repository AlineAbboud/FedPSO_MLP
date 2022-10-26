import os
import numpy as np
from build_model import Model
import csv
import random
from keras.distribute import distributed_training_utils_v1
from sklearn.neural_network import MLPRegressor

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, max_error



import time

# client config
NUMOFCLIENTS = 5 # number of client(as particles)
EPOCHS = 3 # number of total iteration
CLIENT_EPOCHS = 5 # number of each client's iteration
BATCH_SIZE = 10 # Size of batches to train on
ACC = 0.3 # 0.4
LOCAL_ACC = 0.7 # 0.6
GLOBAL_ACC = 1.4 # 1.0

DROP_RATE = 0 # 0 ~ 1.0 float value


# model config 
LOSS = 'categorical_crossentropy' # Loss function
NUMOFCLASSES = 5 # Number of classes
lr = 0.0025
OPTIMIZER = SGD(lr=lr, momentum=0.9, decay=lr/(EPOCHS*CLIENT_EPOCHS), nesterov=False) # lr = 0.015, 67 ~ 69%


def write_csv(algorithm_name, list):

    file_name = '{name}_CIFAR10_randomDrop_{drop}%_output_LR_{lr}_CLI_{cli}_CLI_EPOCHS_{cli_epoch}_TOTAL_EPOCHS_{epochs}_BATCH_{batch}.csv'
    file_name = file_name.format(drop=DROP_RATE, name=algorithm_name, lr=lr, cli=NUMOFCLIENTS, cli_epoch=CLIENT_EPOCHS, epochs=EPOCHS, batch=BATCH_SIZE)
    f = open(file_name, 'w', encoding='utf-8', newline='')
    wr = csv.writer(f)
    
    for l in list:
        wr.writerow(l)
    f.close()


def init_model(train_data_shape):

    model = Model(loss=LOSS, optimizer=OPTIMIZER, classes=NUMOFCLASSES)
    init_model = model.fl_paper_model(train_shape=train_data_shape)
    return init_model


def client_data_config(x_train, y_train):
    client_data = [() for _ in range(NUMOFCLIENTS)] # () for _ in range(NUMOFCLIENTS)
    num_of_each_dataset = int(x_train.shape[0] / NUMOFCLIENTS)

    for i in range(NUMOFCLIENTS):
        split_data_index = []
        while len(split_data_index) < num_of_each_dataset:
            item = random.choice(range(x_train.shape[0]))
            if item not in split_data_index:
                split_data_index.append(item)
        
        new_x_train = np.asarray([x_train[k] for k in split_data_index])
        new_y_train = np.asarray([y_train[k] for k in split_data_index])

        client_data[i] = (new_x_train, new_y_train)

    return client_data

class particle():
    def __init__(self, particle_num, client, x_train, y_train):
        # for check particle id
        self.particle_id = particle_num
        
        # particle model init
        self.particle_model = client
        
        # best model init
        self.local_best_model = client
        self.global_best_model = client

        # best score init
        self.local_best_score = 0.0
        self.global_best_score = 0.0

        self.x = x_train
        self.y = y_train

        # acc = acceleration
        self.parm = {'acc':ACC, 'local_acc':LOCAL_ACC, 'global_acc':GLOBAL_ACC}
        
        # velocities init
        self.velocities = [None] * len(client.coefs_)#get_weights())
        for i, layer in enumerate(client.coefs_):
            self.velocities[i] = np.random.rand(*layer.shape) / 5 - 0.10

    def load_weights(self, filepath, by_name=False):
        self._original_model.load_weights(filepath, by_name=False)

        orig_model_weights = self._original_model.get_weights()
        distributed_training_utils_v1.set_weights(
            self._original_model._distribution_strategy,
            self,
            orig_model_weights,
        )


    def train_particle(self):
        print("particle {}/{} fitting".format(self.particle_id+1, NUMOFCLIENTS))

        # set each epoch's weight
        step_model = self.particle_model
        step_weight = step_model.coefs_ #get_weights()
        
        # new_velocities = [None] * len(step_weight)
        new_weight = [None] * len(step_weight)
        local_rand, global_rand = random.random(), random.random()

        # PSO algorithm applied to weights
        for index, layer in enumerate(step_weight):
            new_v = self.parm['acc'] * self.velocities[index]
            new_v = new_v + self.parm['local_acc'] * local_rand * (self.local_best_model.coefs_[index] - layer)
            new_v = new_v + self.parm['global_acc'] * global_rand * (self.global_best_model.coefs_[index] - layer)
            self.velocities[index] = new_v
            new_weight[index] = step_weight[index] + self.velocities[index]

        # step_model.set_weights(new_weight)
        step_model.coefs_ = new_weight

        
        save_model_path = 'checkpoint/checkpoint_particle_{}'.format(self.particle_id)
        mc = ModelCheckpoint(filepath=save_model_path, 
                            monitor='val_loss', 
                            mode='min',
                            save_best_only=True,
                            save_weights_only=True,
                            )

        print('len 2 = ', len(self.x))
        hist = step_model.fit(self.x, self.y)

        print('hist', hist.best_loss_)
        train_score_loss = hist.best_loss_ #['val_loss'][-1]

        self.particle_model = step_model

        # if self.global_best_score <= train_score_acc:
        if self.global_best_score >= train_score_loss:
            self.local_best_model = step_model
            
        return self.particle_id, train_score_loss
    
    def update_global_model(self, global_best_model, global_best_score):
        if self.local_best_score > global_best_score:
            self.global_best_model = global_best_model
            self.global_best_score = global_best_score

    def resp_best_model(self, gid):
        if self.particle_id == gid:
            return self.particle_model


def get_best_score_by_loss(step_result):
    temp_score = 100000
    temp_index = 0

    for index, result in enumerate(step_result):
        if temp_score > result[1]:
            temp_score = result[1]
            temp_index = index

    return step_result[temp_index][0], step_result[temp_index][1]


def get_best_score_by_acc(step_result):
    temp_score = 0
    temp_index = 0

    for index, result in enumerate(step_result):
        if temp_score < result[1]:
            temp_score = result[1]
            temp_index = index

    return step_result[temp_index][0], step_result[temp_index][1]

def generateTrainingData(floor):
    import pandas as pd
    dataset_folder = os.path.join(os.path.dirname(__file__), "./_data/csv")
    print(dataset_folder + "/data-floor-" + str(floor) +".csv")
    dataset = pd.read_csv(dataset_folder + "/data-floor-" + str(floor) +".csv")

    energy_consumption = dataset["tot_energy"]
    data = dataset.drop("tot_energy", axis=1)

    X_train, X_test, y_train, y_test = train_test_split(data, energy_consumption, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":

    i = 0
    client_data = [() for _ in range(NUMOFCLIENTS)]
    loop_max_err = []
    loop_mae = []
    Epoch_mae = []
    loop_mse = []
    loop_rmse = []

    while i < NUMOFCLIENTS:
        x_train, x_test, y_train, y_test = generateTrainingData(i + 3)
        client_data[i] = (x_train, y_train)
        i = i + 1

    x_train_arr = np.array(x_train)

    server_model = [() for _ in range(NUMOFCLIENTS)]
    loss_curves = [() for _ in range(NUMOFCLIENTS)]

    percentage = 0.25
    for i in range(NUMOFCLIENTS):
        server_model[i] = MLPRegressor(random_state=0, max_iter=10000, batch_size=200)
        server_model[i].fit(client_data[i][0], client_data[i][1])
        loss_curves[i] = server_model[i].loss_curve_

    #
    pso_model = []
    for i in range(NUMOFCLIENTS):
        pso_model.append(particle(particle_num=i, client=server_model[i], x_train=client_data[i][0], y_train=client_data[i][1]))

    server_evaluate_acc = []
    global_best_model = None
    global_best_score = 0.0

    for epoch in range(EPOCHS):
        server_result = []
        start = time.time()
        i=0
        for client in pso_model:
            if epoch != 0:
                client.update_global_model(server_model, global_best_score)

            pid, train_score = client.train_particle()
            rand = random.randint(0,99)
    #
            # Randomly dropped data sent to the server
            drop_communication = range(DROP_RATE)
            if rand not in drop_communication:
                server_result.append([pid, train_score])
        i = i + 1
        # Send the optimal model to each client after the best score comparison
        gid, global_best_score = get_best_score_by_loss(server_result)
        for client in pso_model:
            if client.resp_best_model(gid) != None:
                global_best_model = client.resp_best_model(gid)

        server_model = global_best_model

        y_pred = server_model.predict(x_test)
        Epoch_mae.append(mean_absolute_error(y_test, y_pred))


        loss_cur = server_model.loss_curve_