import pandas as pd, numpy as np, math, matplotlib.pyplot as plt
from json import loads, dumps
from tqdm import tqdm

percision = 5
 
def sigmoid(x):
  return 1 / (1 + np.exp(-x.clip(-15, 15)))

proprietes = {}
data_set = pd.DataFrame()
answers = pd.Series()

learning_data = pd.DataFrame(columns = ["Generation", "Loss", "Testing"])

last_gens = 0

def edit_desc(bar, gen, gens, loss, test):
    global learning_data
    global last_gens
    bar.desc = f"Processing dataset (Loss: {round(float(loss)*10, percision)}, Gen: {gen}/{gens}, Latest test: {test}%, current intensity: {round(calculate_intensity(gen, loss), 4)}x.) "
    
    if len(learning_data) != 0  and gen == 0 and learning_data.loc[len(learning_data)-1, "Generation"]-last_gens != 0:
        last_gens += learning_data.loc[len(learning_data)-1, "Generation"]
        print("added", gen, last_gens, learning_data.loc[len(learning_data)-1, "Generation"])

    learning_data.loc[len(learning_data)] = {"Generation": gen+last_gens, "Loss": (loss*100), "Testing": test}

def plot():
    global learning_data
    plot_data = learning_data.set_index("Generation", drop= True)
    plot_data.plot(ylabel= "Sucsses Rate", yticks= range(101)[::10])
    plt.axhline(50, color = "r", linestyle= "--")
    plt.show()

def calculate_intensity(gen, loss):
    intensity_prop = proprietes["Intensity"]
    mode = intensity_prop[2]
    max_inten = intensity_prop[0]

    if mode != "static":
        min_inten = intensity_prop[1]

    if mode == "gen":
        steps = proprietes["Generations"]
        inten_step = (max_inten - min_inten) / steps
        return inten_step * (steps - gen)
    elif mode == "loss":
        steps = intensity_prop[3] / 10
        inten_step = (max_inten - min_inten) / steps
        if (inten_step * loss) == abs(inten_step * loss):
            return inten_step * loss
        else:
            return max_inten
    elif mode == "static":
        return max_inten


def get_proprietes():
    global proprietes
    proprietes = {}
    with open("Network_Proprietes.json") as file:
        proprietes = file.read()
    proprietes = loads(proprietes)
    
def get_dataset():
    global answers
    global data_set
    data_set = pd.DataFrame()
    answers = pd.Series()

    data_set = pd.read_csv(proprietes["Data Set"] + ".csv")
    if not proprietes["Data set size (0 => all)"] == 0:
        data_set = data_set.head(proprietes["Data set size (0 => all)"])
    data_set.drop(columns= proprietes["Drop columns"], inplace= True)
    answers = data_set[proprietes["Output"]].copy()
    answers = np.ceil(answers.clip(0, 1))
    data_set.drop(columns= proprietes["Output"], inplace= True)
    data_set.reset_index(inplace= True, drop= True)
    data_set = data_set.transpose()
    data_set.reset_index(inplace= True, drop= True)


def reset_plot_data():
    global learning_data
    global last_gens
    last_gens = 0
    learning_data = pd.DataFrame(columns = ["Generation", "Loss", "Testing"])
    
class Layer:
    def __init__(self, num_inputs, num_neurons, hidden= True):
        self.weights = pd.DataFrame(np.random.randn(num_inputs, num_neurons))
        self.biases = pd.DataFrame(np.zeros((1, num_neurons)))
        self.hidden = hidden

    def forward(self, input, answers= pd.Series(), learn= False):
        self.output = (self.weights.transpose()).dot(input.transpose())
        self.output = self.output.transpose() + (self.biases.values.tolist()[0])
        if self.hidden:
            self.output.clip(0, inplace= True)
        else:
            self.output = sigmoid(self.output)
            if learn and not answers.empty:
                self.output = self.output[pd.get_dummies(answers.transpose())].transpose().max().mean()
            elif not answers.empty:
                self.output = self.output.max(axis= 1) == self.output[pd.get_dummies(answers.transpose())].max(axis= 1)

    def update(self, gen, loss):
        self.weights += pd.DataFrame(np.random.randn(self.weights.shape[0], self.weights.shape[1]) * calculate_intensity(gen, loss))
        self.biases += pd.DataFrame(np.random.randn(self.biases.shape[0], self.biases.shape[1]) * calculate_intensity(gen, loss))
        
layers = []
def init_layers():
    global layers
    layers = []
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
    
def run_batch(batch_num, batch_count, lowest_loss, gen):
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
                for l in layers:
                    load_layers()
                    l.update(gen, lowest_loss)
                return None
            else:
                return layer.output
            
def learn():
    generations = proprietes["Generations"]
    batch_count = math.floor(data_set.shape[1] / proprietes["Batch Size"])
    lowest_loss = 1
    latest_test = test()
    pbar = tqdm(desc="Processing dataset", colour='#0997FF', total= (generations * (batch_count - proprietes["Test batch count"])))
    for generation in range(generations):
        for batch in range(batch_count - proprietes["Test batch count"]):
            edit_desc(pbar, generation, generations, lowest_loss, latest_test)
            pbar.update(1)
            save_layers()
            loss = run_batch(batch, batch_count, lowest_loss, generation)

            if loss is not None:
                lowest_loss = loss

            if round(lowest_loss, percision) == 0:                
                edit_desc(pbar, generation, generations, lowest_loss, test())

                pbar = None

                print(f"We reached loss = 0.0 So the learning proccess doesn't need to continue. (We were on gen: {generation+1}/{generations}. and on batch {batch+1}/{batch_count - proprietes['Test batch count']}.)")
                
                return False
            
            elif proprietes["Test frequency (0 => Every gen.)"] == 0:
                if batch == 0:
                    latest_test = test()
                    edit_desc(pbar, generation, generations, lowest_loss, latest_test)

            elif ((generation+1)*batch) % proprietes["Test frequency (0 => Every gen.)"] == 0:
                latest_test = test()
                edit_desc(pbar, generation, generations, lowest_loss, latest_test)


    latest_test = test()
    edit_desc(pbar, generations, generations, lowest_loss, latest_test)
    return True

def test():
    batch_count = math.floor(data_set.shape[1] / proprietes["Batch Size"])

    X = data_set.transpose().loc[(data_set.transpose().index % batch_count) >= batch_count - proprietes["Test batch count"] ].copy()
    X.reset_index(inplace= True, drop= True)

    y = answers.loc[(answers.index % batch_count) >= (batch_count - proprietes["Test batch count"])].copy()
    y.reset_index(inplace= True, drop= True)

    final = 0

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
            final = layer.output

            final = final.sum()/final.size

    return round(final*100, 2)


while True:
    print("\n---\n")

    command = input("Console -> ")
    command = command.lower()

    print()

    if command in ["run", "r"]:
        get_proprietes()
        get_dataset()
        init_layers()
        reset_plot_data()
        learn()
    elif command in ["learn", "l"]:
        learn()
    elif command in ["plot", "p"]:
        plot()
    elif command in ["get prop", "gp"]:
        get_proprietes()
        print("Proprietes were loaded sucsessfuly.")
    elif command in ["reload", "r"]:
        init_layers()
        print("Layers were initilazed sucsessfuly.")
    elif command in ["update data", "uds"]:
        get_dataset()
        print("Data set was loaded sucsessfuly.")

    elif command in ["help", "h"]:
        print("All commands:\n - run (Resers everithing and learns.)\n - learn/l (Starts learning process.)\n - loop learn;'and a number of iterations'/ll;'and a number of iterations' (Runs learn many times.)\n - plot/p (Plots loss and tests against generations.)\n - get prop/gp (Reloads/loads proprietes.)\n - reload/r (Creates/resets the neural network.)\n - update data/uds (Gets data from data set.)\n - prep/pr (Prepear for learning.)")
    elif command.startswith("loop learn;") or command.startswith("ll;"):
        for i in range(int(command[command.index(";")+1:])):
            print(f"\n - We are running learn process for {i+1}. time:")
            learn()

    elif command in ["prep", "pr"]:
        get_proprietes()
        print("Proprietes were loaded sucsessfuly.")
        get_dataset()
        print("Layers were initilazed sucsessfuly.")
        init_layers()
        print("Data set was loaded sucsessfuly.")

    elif command in ["reset plot", "rp"]:
        reset_plot_data()

    else:
        print("Error: non existing command. (Type: Help)")