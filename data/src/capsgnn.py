import glob
import json
import torch
import random
import numpy as np
import pandas as pd
from tqdm import tqdm, trange
from torch_geometric.nn import GCNConv
from utils import create_numeric_mapping
from layers import ListModule, PrimaryCapsuleLayer, Attention, SecondaryCapsuleLayer, margin_loss, TaskRouter

class CapsGNN(torch.nn.Module):
    """
    An implementation of themodel described in the following paper:
    https://openreview.net/forum?id=Byl8BnRcYm
    """
    def __init__(self, args, number_of_features, number_of_targets,active_task=0):
        super(CapsGNN, self).__init__()
        """
        :param args: Arguments object.
        :param number_of_features: Number of vertex features.
        :param number_of_targets: Number of classes.
        """
        self.args = args
        self.number_of_features = number_of_features
        self.number_of_targets = number_of_targets
        self.number_of_tasks = 8
        self._setup_layers()

    def set_active_task(self, active_task):
        self.active_task = active_task
        return active_task
    
    def _setup_base_layers(self):
        """
        Creating GCN layers.
        """
        self.base_layers = [GCNConv(self.number_of_features, self.args.gcn_filters)]
        for layer in range(self.args.gcn_layers-1):
            self.base_layers.append(GCNConv( self.args.gcn_filters, self.args.gcn_filters))
        self.base_layers = ListModule(*self.base_layers)

    def _setup_primary_capsules(self):
        """
        Creating primary capsules.
        """
        self.first_capsule = PrimaryCapsuleLayer(in_units = self.args.gcn_filters, in_channels = self.args.gcn_layers, num_units = self.args.gcn_layers, capsule_dimensions = self.args.capsule_dimensions)

    def _setup_attention(self):
        """
        Creating attention layer.
        """
        self.attention = Attention(self.args.gcn_layers* self.args.capsule_dimensions, self.args.inner_attention_dimension)

    def _setup_graph_capsules(self):
        """
        Creating graph capsules.
        """
        self.graph_capsule = SecondaryCapsuleLayer(self.args.gcn_layers, self.args.capsule_dimensions, self.args.number_of_capsules, self.args.capsule_dimensions)

    def _setup_class_capsule(self):
        """
        Creating class capsules.
        """

        
        for i in range(self.number_of_tasks):
            vars(self)["class_capsule_%s"%i] = SecondaryCapsuleLayer(self.args.capsule_dimensions,
                                                                     self.args.number_of_capsules, 
                                                                     self.number_of_targets,
                                                                     self.args.capsule_dimensions)
            
        
        
        

    def _setup_reconstruction_layers(self):
        """
        Creating histogram reconstruction layers.
        """
        
        for i in range(self.number_of_tasks):
            
            vars(self)["reconstruction_layer_1_%s"%i] = torch.nn.Linear(self.number_of_targets*self.args.capsule_dimensions, int((self.number_of_features * 2) / 3))
            
            vars(self)["reconstruction_layer_2_%s"%i] = torch.nn.Linear(int((self.number_of_features * 2) / 3), int((self.number_of_features * 3) / 2))
            
            vars(self)["reconstruction_layer_3_%s"%i] = torch.nn.Linear(int((self.number_of_features * 3) / 2), self.number_of_features)
            
            
            

    
    def _setup_taskrouting_layers(self):
        '''
        Creating task routing layers
        '''
        
        self.taskrouting = TaskRouter(unit_count = self.number_of_features, task_count=self.number_of_tasks, sigma= 0.5, name="TaskRouter")
        
        
    def _setup_layers(self):
        """
        Creating layers of model.
        1. GCN layers.
        2. Primary capsules.
        3. Attention
        4. Graph capsules.
        5. Class capsules.
        6. Reconstruction layers.
        """
        self._setup_base_layers()
        self._setup_primary_capsules()
        self._setup_attention()
        self._setup_graph_capsules()
        self._setup_class_capsule()
        self._setup_reconstruction_layers()
        
        self._setup_taskrouting_layers(self)

    def calculate_reconstruction_loss(self, capsule_input, features,task):
        """
        Calculating the reconstruction loss of the model.
        :param capsule_input: Output of class capsule.
        :param features: Feature matrix.
        :return reconstrcution_loss: Loss of reconstruction.
        """

        v_mag = torch.sqrt((capsule_input**2).sum(dim=1))
        _, v_max_index = v_mag.max(dim=0)
        v_max_index = v_max_index.data

        capsule_masked = torch.autograd.Variable(torch.zeros(capsule_input.size()))
        capsule_masked[v_max_index,:] = capsule_input[v_max_index,:]
        capsule_masked = capsule_masked.view(1, -1)

        feature_counts = features.sum(dim=0)
        feature_counts = feature_counts/feature_counts.sum()

        
        reconstruction_output = torch.nn.functional.relu(vars(self)["reconstruction_layer_1_%s"%task](capsule_masked))
        reconstruction_output = torch.nn.functional.relu(vars(self)["reconstruction_layer_2_%s"%task](reconstruction_output))
        reconstruction_output = torch.softmax(vars(self)["reconstruction_layer_3_%s"%task](reconstruction_output),dim=1)
        reconstruction_output = reconstruction_output.view(1, self.number_of_features)

        reconstruction_loss = torch.sum((features-reconstruction_output)**2)

        
        return reconstruction_loss
        
    def forward(self, data):
        """
        Forward propagation pass.
        :param data: Dictionary of tensors with features and edges.
        :return class_capsule_output: Class capsule outputs.
        """
        features = data["features"]
        edges = data["edges"]
        hidden_representations = []
        
        for layer in self.base_layers:
            features = torch.nn.functional.relu(layer(features, edges))
            hidden_representations.append(features)

        hidden_representations = torch.cat(tuple(hidden_representations))
        hidden_representations = hidden_representations.view(1, self.args.gcn_layers, self.args.gcn_filters,-1)
        first_capsule_output = self.first_capsule(hidden_representations)
        first_capsule_output = first_capsule_output.view(-1,self.args.gcn_layers* self.args.capsule_dimensions)
        
        rescaled_capsule_output = self.attention(first_capsule_output)
        rescaled_first_capsule_output = rescaled_capsule_output.view(-1, self.args.gcn_layers, self.args.capsule_dimensions)
        graph_capsule_output = self.graph_capsule(rescaled_first_capsule_output)
        reshaped_graph_capsule_output = graph_capsule_output.view(-1, self.args.capsule_dimensions, self.args.number_of_capsules) 
        
        
        ######
        output = tuple()
        for i in range(self.number_of_tasks):
            vars()["class_capsule_output_%s"%i] = vars(self)["class_capsule_%s"%i](reshaped_graph_capsule_output)
            
            vars()["class_capsule_output_%s"%i] = vars()["class_capsule_output_%s"%i].view(-1,
                                                                self.number_of_targets*self.args.capsule_dimensions )
            
            vars()["class_capsule_output_%s"%i] = torch.mean(vars()["class_capsule_output_%s"%i],dim=0).view(1,
                                                                self.number_of_targets,self.args.capsule_dimensions)
            
            output = output + (vars()["class_capsule_output_%s"%i],)
        
        
        #######
        reconstruction_loss = 0
        for i in range(self.number_of_tasks):
            vars()['recon_%s'%i] = vars()["class_capsule_output_%s"%i].view(
                                            self.number_of_targets,self.args.capsule_dimensions)
            
            vars()["reconstruction_loss_%s"%i] = self.calculate_reconstruction_loss(
                                                vars()['recon_%s'%i],data["features"],i)
            
            reconstruction_loss+= vars()["reconstruction_loss_%s"%i]
            
            output = output + (vars()["reconstruction_loss_%s"%i],)

        #######
        
        router = self.taskrouting(unit_count = self.number_of_features, task_count=self.number_of_tasks, sigma= 0.5, name="TaskRouter")
        
        
        return output

                                        
class CapsGNNTrainer(object):
    """
    CapsGNN training and scoring.
    """
    def __init__(self,args):
        """
        :param args: Arguments object.
        """
        self.args = args
        self.setup_model()

    def enumerate_unique_labels_and_targets(self):
        """
        Enumerating the features and targets in order to setup weights later.
        """
        print("\nEnumerating feature and target values.\n")
        ending = "*.json"

        self.train_graph_paths = glob.glob(self.args.train_graph_folder+ending)
        self.test_graph_paths = glob.glob(self.args.test_graph_folder+ending)
    
        graph_paths = self.train_graph_paths + self.test_graph_paths
        
        #print(graph_paths)
        
        targets = set()
        features = set()
        for path in tqdm(graph_paths):
            data = json.load(open(path))
            targets = targets.union(set([data["target0"]]))
            features = features.union(set(data["labels"]))

        
        
        self.target_map = create_numeric_mapping(targets)
        self.feature_map = create_numeric_mapping(features)

        self.number_of_features = len(self.feature_map)
        self.number_of_targets = len(self.target_map)
    
    def setup_model(self):
        """
        Enumerating labels and initializing a CapsGNN.
        """
        self.enumerate_unique_labels_and_targets()
        self.model = CapsGNN(self.args, self.number_of_features, self.number_of_targets)

    def create_batches(self):
        """
        Batching the graphs for training.
        """
        self.batches = [self.train_graph_paths[i:i + self.args.batch_size] for i in range(0,len(self.train_graph_paths), self.args.batch_size)]

    def create_data_dictionary(self, target, edges, features):
        """
        Creating a data dictionary.
        :param target: Target vector.
        :param edges: Edge list tensor.
        :param features: Feature tensor.
        """
        to_pass_forward = dict()
        for i in range(len(target)):
            to_pass_forward["target%s"%i] = target[i]
        
        to_pass_forward["edges"] = edges
        to_pass_forward["features"] = features
        return to_pass_forward

    def create_target(self, data,task):
        """
        Target createn based on data dicionary.
        :param data: Data dictionary.
        :return : Target vector.
        """
        return  torch.FloatTensor([0.0 if i != data["target%s"%task] else 1.0 for i in range(self.number_of_targets)])

    def create_edges(self,data):
        """
        Create an edge matrix.
        :param data: Data dictionary.
        :return : Edge matrix.
        """
        edges = [[edge[0],edge[1]] for edge in data["edges"]] + [[edge[1],edge[0]] for edge in data["edges"]]
        return torch.t(torch.LongTensor(edges))

    def create_features(self,data):
        """
        Create feature matrix.
        :param data: Data dictionary.
        :return features: Matrix of features.
        """
        features = np.zeros((len(data["labels"]), self.number_of_features))
        node_indices = [node for node in range(len(data["labels"]))]
        feature_indices = [self.feature_map[label] for label in data["labels"].values()] 
        features[node_indices,feature_indices] = 1.0
        features = torch.FloatTensor(features)
        return features

    def create_input_data(self, path,task):
        """
        Creating tensors and a data dictionary with Torch tensors.
        :param path: path to the data JSON.
        :return to_pass_forward: Data dictionary.
        """
        data = json.load(open(path))
        target = []
        for i in range(task):
            target.append(self.create_target(data,i))
        edges = self.create_edges(data)
        features = self.create_features(data)
        to_pass_forward = self.create_data_dictionary(target, edges, features)
        return to_pass_forward

        
        
    def fit(self):
        """
        Training a model on the training set.
        """
        print("\nTraining started.\n")
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)

        for epoch in tqdm(range(self.args.epochs), desc = "Epochs: ", leave = True):
            random.shuffle(self.train_graph_paths)
            self.create_batches()
            losses = 0       
            self.steps = trange(len(self.batches), desc="Loss")
            for step in self.steps:
                accumulated_losses = 0
                optimizer.zero_grad()
                batch = self.batches[step]
                for path in batch:
                    task = 8
                    loss = 0
                    data = self.create_input_data(path,task)
                    batch_output = self.model(data)
                                        
                    for i in range(task):
                        loss+=margin_loss(batch_output[i],data["target%s"%i],self.args.lambd)
                        loss+=self.args.theta*batch_output[i+task]
                    
       
                    
                    accumulated_losses = accumulated_losses + (loss/task)
                
                accumulated_losses = accumulated_losses/len(batch)
                accumulated_losses.backward()
                optimizer.step()
                losses = losses + accumulated_losses.item()
                average_loss = losses/(step + 1)
                self.steps.set_description("CapsGNN (Loss=%g)" % round(average_loss,4))

    def score(self):
        """
        Scoring on the test set.
        """
        print("\n\nScoring.\n")
        self.model.eval()
        #self.predictions = []
        #self.hits = []
        task = 8
        for i in range(task):
            vars(self)["predictions_%s"%i] = []
            vars(self)["hits_%s"%i] = []
            
        for path in tqdm(self.test_graph_paths):
            #####
            data = self.create_input_data(path,task)
            results = self.model(data)
            #####
            for i in range(task):
            
                prediction_mag = torch.sqrt((results[i]**2).sum(dim=2))
                _, prediction_max_index = prediction_mag.max(dim=1)
                prediction = prediction_max_index.data.view(-1).item()
                
                vars(self)["predictions_%s"%i].append(prediction)
                vars(self)["hits_%s"%i].append(data["target%s"%i][prediction]==1.0)
                
                #self.predictions.append(prediction)
                #self.hits.append(data["target1"][prediction]==1.0)
        
        acc_outfile = open('./output/accuracy.txt','w')
        for i in range(task):
            acc_outfile.write("Accuracy%s: "%i + str(round(np.mean(vars(self)["hits_%s"%i]),4)) + '\n') 
            print("\nAccuracy%s: "%i + str(round(np.mean(vars(self)["hits_%s"%i]),4)))
            #print("\nAccuracy: " + str(round(np.mean(self.hits),4)))
        
        acc_outfile.close()
    
    def save_predictions(self):
        """
        Saving the test set predictions.
        """
        task = 8
        identifiers = [path.split("/")[-1].strip(".json") for path in self.test_graph_paths]
        torch.save(self.model,'./output/model.pt')
        for i in range(task):
            out = pd.DataFrame()
            out["id"] = identifiers
            out["predictions"] = vars(self)["predictions_%s"%i]
            out["hits"] = vars(self)["hits_%s"%i]
            #out["predictions"] = self.predictions_0
            #out.to_csv(self.args.prediction_path, index = None)
            out.to_csv("./output/predictions_%s"%i, index = None)
