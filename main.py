import pandas as pd, numpy as np, math
from json import loads, dumps
from tqdm import tqdm

e = math.exp(1)

proprietes = {}
data_set = pd.DataFrame()
answers = pd.Series()

def get_proprietes():
    global proprietes
    with open("Network_Proprietes.json") as file:
        proprietes = file.read()
    proprietes = loads(proprietes)
    
def get_dataset():
    global answers
    global data_set
    data_set = pd.read_csv(proprietes["Data Set"] + ".csv")
    if not proprietes["Data set size (0 => all)"] == 0:
        data_set = data_set.head(proprietes["Data set size (0 => all)"])
    data_set.drop(columns= proprietes["Drop colunms"], inplace= True)
    answers = data_set[proprietes["Output"]].copy()
    answers = np.ceil(answers.clip(0, 1))
    data_set.drop(columns= proprietes["Output"], inplace= True)
    data_set.reset_index(inplace= True, drop= True)
    data_set = data_set.transpose()
    data_set.reset_index(inplace= True, drop= True)
    
class Layer:
    def __init__(self, num_inputs, num_neurons, hidden= True):
        self.weights = pd.DataFrame(np.random.randn(num_inputs, num_neurons))
        self.biases = pd.DataFrame(np.zeros((1, num_neurons)))
        self.hidden = hidden

    def forward(self, input, answers= [], learn= False):
        self.output = (self.weights.transpose()).dot(input.transpose())
        self.output = self.output.transpose() + (self.biases.values.tolist()[0])
        if self.hidden:
            self.output.clip(0, inplace= True)
        else:
            self.output = self.output.transpose()
            self.output = e ** (self.output.transpose() - self.output.max(axis= 1))
            self.output = (self.output.transpose() / self.output.sum(axis= 1)).T
            if learn and not answers.empty:
                avg = self.output[pd.get_dummies(answers.transpose())].transpose().max().mean()
                self.output = 1 - avg
            elif not answers.empty:
                self.output = self.output.max(axis= 1) == self.output[pd.get_dummies(answers.transpose())].transpose().max()


    def update(self):
        self.weights += pd.DataFrame(np.random.randn(self.weights.shape[0], self.weights.shape[1]) * proprietes["Intensity"])
        self.biases += pd.DataFrame(np.random.randn(self.biases.shape[0], self.biases.shape[1]) * proprietes["Intensity"])
        
layers = []
def init_layers():
    inputs = proprietes["Input"]
    for neuron_count, idx in zip(proprietes["Layers"], range(len(proprietes["Layers"]))):
        if not idx+1 == len(proprietes["Layers"]):
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
    
def run_batch(batch_num, batch_count, lowest_loss):
    X = data_set.transpose().loc[(data_set.transpose().index % batch_count) == batch_num].copy()
    y = answers.loc[(answers.index % batch_count) == batch_num].copy()
    X.reset_index(inplace= True, drop= True)
    y.reset_index(inplace= True, drop= True)
    output = []
    for layer in layers:
        idx = layers.index(layer)
        if idx == 0:
            layer.forward(X)
            output = layer.output
        elif not idx+1 == len(layers):
            layer.forward(output)
            output = layer.output
        else:
            layer.forward(output, answers= y, learn= True)
            if layer.output >= lowest_loss:
                layer.update()
                return None
            else:
                return layer.output
            
def learn():
    generations = proprietes["Generations"]
    batch_count = math.floor(data_set.shape[1] / proprietes["Batch Size"])
    lowest_loss = 1
    pbar = tqdm(desc="Processing dataset", colour='#0997FF', total= (generations * (batch_count - proprietes["Test batch count"])))
    for generation in range(generations):
        for batch in range(batch_count - proprietes["Test batch count"]):
            pbar.desc = f"Processing dataset (Loss: {round(lowest_loss, 6)}, Gen: {generation}/{generations})"
            pbar.update(1)
            save_layers()
            loss = run_batch(batch, batch_count, lowest_loss)
            if loss is not None:
                lowest_loss = loss
                load_layers()
    pbar.desc = f"Processing dataset (Loss: {round(lowest_loss, 6)}, Gen: {generations}/{generations})"
    return lowest_loss

def test():
    batch_count = math.floor(data_set.shape[1] / proprietes["Batch Size"])

    X = data_set.transpose().loc[(data_set.transpose().index % batch_count) >= batch_count - proprietes["Test batch count"] ].copy()
    X.reset_index(inplace= True, drop= True)

    y = answers.loc[(answers.index % batch_count) >= batch_count - proprietes["Test batch count"]].copy()
    y.reset_index(inplace= True, drop= True)

    output = []
    for layer in layers:
        idx = layers.index(layer)
        if idx == 0:
            layer.forward(X)
            output = layer.output
        elif not idx+1 == len(layers):
            layer.forward(output)
            output = layer.output
        else:
            layer.forward(output, y)
            output = layer.output
            return output.sum()/output.size


get_proprietes()
get_dataset()
init_layers()
learn()
#test()
print(test())