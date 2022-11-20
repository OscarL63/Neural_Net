import random 
import numpy as np

class Node():

    def __init__(self):
        self.v = 0
        self.x_out = 0
        self.bias = 0
        self.weights = [] # incoming weights
        self.delta = 0

    def set_weight_init(self, weight): # set the weight initially, as lists is originally empty 
        self.weights.append(weight)
    
    def set_weight(self, index, weight):
        self.weights[index] = weight

    def get_weight(self, index):
        return self.weights[index]

    def get_weights(self):
        return self.weights

    def set_x_out(self, x_out):
        self.x_out = x_out

    def get_x_out(self):
        return self.x_out

    def set_v(self, v):
        self.v = v
    
    def get_v(self):
        return self.v
    
    def set_bias(self, bias):
        self.bias = bias

    def get_bias(self):
        return self.bias

    def set_delta(self, delta):
        self.delta = delta
    
    def get_delta(self):
        return self.delta

class Layer():

    def __init__(self, node_num, activation_func):


        self.nodes = []
        self.node_num = node_num
        self.activation_func = activation_func

        #print(node_num)

        for node in range(node_num):
            self.nodes.append(Node())
        
        #print(self.nodes)
    
    def get_node(self, node_ind):
        return self.nodes[node_ind]

    def get_nodes(self):
        return self.nodes


class Net():

    def __init__(self, network_size, activation_func_list):

        self.network = []
        self.network_size = network_size
        self.activation_func_list = activation_func_list

        for i in range(len(network_size)):
            layer = Layer(network_size[i], activation_func_list[i])
            self.network.append(layer)

    def populate_weights(self):

        for layer_ind in range(len(self.network) - 1):
            layer = self.network[layer_ind + 1] # we dont need to initialise the input layer weights

            num_of_nodes = self.network_size[layer_ind + 1]

            for node_ind in range(num_of_nodes):
                num_weight = self.network_size[layer_ind] # get the number of nodes from the previous layer

                node = layer.get_node(node_ind)
                for weight in range(num_weight):

                    random_weight = random.uniform(-1,1)
                    node.set_weight_init(random_weight)
        
    def input_data(self, data):

        input_layer = self.network[0]

        for node_ind in range(self.network_size[0]):
            node = input_layer.get_node(node_ind)
            node.set_x_out(data[node_ind])

    def get_output(self):

        output_layer = self.network[-1]

        output_val = []

        for node in output_layer.get_nodes():
            output_val.append(node.get_x_out())

        return output_val

    def activation_function(self, val, activation_func):

        if activation_func == "lin":
            return val

        elif activation_func == "tanh":
            return np.tanh(val) 
    
    def diff_activation_function(self, val, activation_func):

        if activation_func == "lin":
            return 1

        elif activation_func == "tanh":
            output = 1 - self.activation_function(val, "tanh") ** 2
            return output
            

    def forward_pass(self):
        for layer_ind in range(len(self.network) - 1): # we dont want to use the last layer

            layer = self.network[layer_ind + 1] # +1 as layer_ind goes from 0,1,2 and we want the 1,2,3 layer
            prev_layer = self.network[layer_ind] 

            num_of_nodes = self.network_size[layer_ind + 1]
            prev_layer_node = self.network_size[layer_ind]


            for node_ind in range(num_of_nodes):
                node = layer.get_node(node_ind)
                weights = node.get_weights()

                current_node_v = 0

                for prev_node_ind in range(prev_layer_node):
                    prev_weight = weights[prev_node_ind]
                    prev_node_x_out = prev_layer.get_node(prev_node_ind).get_x_out()

                    current_node_v = current_node_v + prev_weight * prev_node_x_out
                #print(current_node_v)

                # Add bias 
                current_node_v = current_node_v + node.get_bias()

                node.set_v(current_node_v)

                # Apply activation function
                current_x_out = self.activation_function(current_node_v, self.activation_func_list[layer_ind + 1])
                node.set_x_out(current_x_out)


    def calculate_out_delta(self, node, desired):

        node_v = node.get_v()
        activation_function = self.activation_func_list[-1]

        error = desired - self.activation_function(node_v, activation_function)
        delta = error * self.diff_activation_function(node_v, activation_function)

        node.set_delta(delta)


    def calculate_hid_delta(self, layer, j):
        
        current_layer = layer - 1 # s
        current_node = self.network[current_layer].get_node(j)
        current_node_v = current_node.get_v()
        #print(current_node)

        delta_layer = self.network[layer]

        delta_sum = 0
        #print("a")
        #print("delta layer ", layer)
        
        for delta_ind in range(len(delta_layer.get_nodes())):
            delta_node = delta_layer.get_node(delta_ind)
            #print("index", len(delta_layer.get_nodes()))

            weight = delta_node.get_weight(j)
            delta_j = delta_node.get_delta()    
            #print("weight", weight)
            #print("delta_j", delta_j)
        

            delta_sum = delta_sum + weight * delta_j
            #print("delta_prog", delta_sum)

        
        cur_activation_func = self.activation_func_list[current_layer]
        
        
        delta = delta_sum * self.diff_activation_function(current_node_v, cur_activation_func)
        current_node.set_delta(delta) 
        
        #print(delta)

    def backward_pass(self, desired_array, learning_rate):

        output_nodes = self.network[-1].get_nodes() # get the last layer nodes

        # Setting the output delta
        for node, desired in zip(output_nodes, desired_array):
            self.calculate_out_delta(node, desired)

        # Loop through each weight and update the weight
        for layer_ind_back in range(len(self.network)-1, 0, -1):
            layer = self.network[layer_ind_back]

            num_of_nodes = self.network_size[layer_ind_back]

            # Loop through every node
            for node_ind in range(num_of_nodes):
                node = layer.get_node(node_ind)

                number_of_weights = len(node.get_weights())

                # Loop through every weight in the nodes
                for weight_ind in range(number_of_weights):
                    
                    # weight_ind = i, node_ind = j
                    prev_layer_x_out = self.network[layer_ind_back - 1].get_node(weight_ind).get_x_out()

                    weight = node.get_weight(weight_ind)
                    update_weight = weight + learning_rate * node.get_delta() * prev_layer_x_out
                    #print("x", prev_layer_x_out)
                    #print("del", node.get_delta())
                    #print("node ", layer_ind_back, node_ind)
                    #print("up", update_weight)
                    #print("ind", weight_ind)
                    node.set_weight(weight_ind, update_weight)
                    #print("------")
                    #print(update_weight)

                # Update the bias here !!!!!!!!!!!!!!!!!!!!!!!!!
                old_bias = node.get_bias()
                new_bias = old_bias + learning_rate * node.get_delta()
                node.set_bias(new_bias)
                
            # Calculate delta with new updated weights once every layer is completed
            # No need to calculate the delta for the input layer
            #print("layer", layer_ind_back)

            # No need to calculate the delta for the input layer
            if layer_ind_back-1 != 0:
                pre_layer = self.network[layer_ind_back - 1]
                
                for pre_layer_node in range(len(pre_layer.get_nodes())):
                    #print(len(pre_layer))
                    #print(pre_layer_node)
                    self.calculate_hid_delta(layer_ind_back, pre_layer_node)

                #self.calculate_hid_delta(layer_ind_back, node_ind)
                #print(layer_ind_back - 1)
                
            #print("-------------------------------------")






# main 
if __name__ == "__main__":
    """ network_size = [4,5,3,3]
    activation_function_list = ["start", "tanh", "tanh","lin"] # always match the size of network size
    net = Net(network_size, activation_function_list)
    net.populate_weights()
    net.input_data([-0.2,0.5,-1.3,0.68])
    net.forward_pass()
    desired_array = [0,0,1]
    learning_rate = 0.01
    net.backward_pass(desired_array, learning_rate) """

    # Hyper parameters
    network_size = [4,5,3,3]
    activation_function_list = ["start", "tanh", "tanh","lin"]
    learning_rate = 0.01
    epochs = 500


    # Create network 
    net = Net(network_size, activation_function_list)
    net.populate_weights()

    # Load Data
    file = open("IrisData.txt", "r")
    data_set = file.read().splitlines() 
    random.shuffle(data_set)

    training_amount = int(0.7 * len(data_set))
    testing_amount = len(data_set) - training_amount

    # Training

    for epoch in range(epochs):
        for data_ind in range(training_amount):
            csv_data = data_set[data_ind].split(",")

            if csv_data[-1] == "Iris-setosa":
                des_arr = [0.6, -0.6, -0.6]

            elif csv_data[-1] == "Iris-versicolor":
                des_arr = [-0.6, 0.6, -0.6]

            elif csv_data[-1] == "Iris-virginica":
                des_arr = [-0.6, -0.6, 0.6]
            
    
            input_data = list(map(lambda a: float(a), csv_data[0:-1]))

            net.input_data(input_data)
            net.forward_pass()
            net.backward_pass(des_arr, learning_rate)

    
    # Testing
    correct = 0

    for index in range(testing_amount):
        test_ind = training_amount + index

        csv_data = data_set[test_ind].split(",")
        input_data = list(map(lambda a: float(a), csv_data[0:-1]))

        net.input_data(input_data)
        net.forward_pass()

        output_guess = net.get_output().index(max(net.get_output()))

        if output_guess == 0 and csv_data[-1] == "Iris-setosa":
            correct = correct + 1

        elif output_guess == 1 and csv_data[-1] == "Iris-versicolor":
            correct = correct + 1

        elif output_guess == 2 and csv_data[-1] == "Iris-virginica":
            correct = correct + 1


    print(correct)





    file.close()

