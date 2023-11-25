import pandas as pd, numpy as np, math
from json import loads, dumps
from tqdm import tqdm

e = math.exp(1)

proprietes = {}
data_set = pd.DataFrame()
ansers = pd.Series()

def get_proprietes():
    global proprietes
    with open("Network_Proprietes.json") as file:
        proprietes = file.read()
    proprietes = loads(proprietes)

def get_dataset():
    global ansers
    global data_set
    data_set = pd.read_csv(proprietes["Data Set"] + ".csv")
    data_set.drop(columns= proprietes["Drop colunms"], inplace= True)
    ansers = data_set[proprietes["Output"]]
    data_set.drop(columns= proprietes["Output"], inplace= True)
    data_set.reset_index(inplace= True, drop= True)
    data_set = data_set.transpose()
    data_set.reset_index(inplace= True, drop= True)

class Layer:
    def __init__(self, num_inputs, num_neurons, hidden= True):
        self.weights = pd.DataFrame(np.random.randn(num_inputs, num_neurons))
        self.biases = pd.DataFrame(np.zeros((1, num_neurons)))
        self.hidden = hidden
    def forward(self, input, ansers= [], learning = True):
        self.output = (self.weights.transpose()).dot(input.transpose())
        self.output = self.output.transpose() + (self.biases.values.tolist()[0])
        if self.hidden:
            self.output.clip(0, inplace= True)
        else:
            self.output = (e ** ((self.output.transpose() - self.output.max(axis= 1)).transpose()))
            self.output = (self.output.transpose() / self.output.sum(axis= 1)).transpose()
            if learning:
                avg = self.output[pd.get_dummies(ansers)].max().mean()
                self.output = 1 - avg
    def update(self):
        self.weights += proprietes["Intensity"] * pd.DataFrame(np.random.randn(self.weights.shape))
        self.biases += proprietes["Intensity"] * pd.DataFrame(np.random.randn(self.biases.shape))

layers = []
def init_layers():
    inputs = proprietes["Inputs"]
    for neuron_count in proprietes["Layers"]:
        idx = proprietes["Layers"].index(neuron_count)
        if not idx == len(proprietes["Layers"]):
            layers.append(Layer(inputs, neuron_count))
            inputs = neuron_count
        else:
            layers.append(Layer(inputs, neuron_count, hidden= False))

layers_save = []

def save_layers():
    global layers_save
    layers_save = layers.copy()

def load_layers():
    global layers
    layers = layers_save.copy()

def run_batch(batch_num, lowest_loss):
    X = data_set.loc[(data_set.index % (data_set.shape[1] % proprietes["Batch Size"])) == batch_num].copy()
    y = ansers.loc[(ansers.index % (data_set.shape[1] % proprietes["Batch Size"])) == batch_num].copy()
    last_layer = 0
    for layer in layers:
        idx = layers.index(layer)
        if idx == 0:
            layer.forward(X.transpose())
            last_layer = layer
        if not idx == len(layers):
            layer.forward(last_layer.output)
            last_layer = layer
        else:
            layer.forward(last_layer.output, ansers= y, learning= True)
            if layer.output > lowest_loss:
                layer.update()
                return None
            else:
                return layer.output
        
def learn():
    generations = proprietes["Generations"]
    batch_count = data_set.shape[1] % proprietes["Batch Size"]
    lowest_loss = 1
    pbar = tqdm(desc="Processing dataset", colour='#0997FF', total= generations * batch_count)
    for generation in range(proprietes["Generations"]):
        for batch in range(proprietes["Batch Size"]):
            pbar.desc = f"Processing dataset (Loss: {round(lowest_loss, 4)}, Gen: {generation+1}/{generations})"
            pbar.update(1)
            save_layers()
            loss = run_batch(batch, lowest_loss)
            if not loss is None:
                lowest_loss = loss
                load_layers()

def run():
    get_proprietes()
    get_dataset()
    init_layers()
    learn()

run()