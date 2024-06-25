import torch
import torch.nn as nn
import sys, os
from torch.autograd import Variable
import numpy as np
import random
import psutil
import subprocess
from tqdm import tqdm
import scipy,math
import time

from .data_utils import *
from .global_utils import *
from .plot_utils import *

SAVE_FORMAT = "pickle"


class GRU():
    def __init__(self, params):
        self.start_time = time.time()
        self.rnnName = 'GRU'
        self.rnnChaosName = self.rnnName + '-' + params['system_name']
        self.params = params.copy()

        # self.train_time_limit = secondsToTimeStr(3600*params['train_time_limit_in_hour'])
        self.train_time_limit = 3600*params['train_time_limit_in_hour']

        self.gpu = torch.cuda.is_available()
        self.device_count = torch.cuda.device_count()
        if self.gpu:
            self.torch_dtype = torch.cuda.DoubleTensor
            torch.set_default_tensor_type(torch.cuda.DoubleTensor)
        else:
            self.torch_dtype = torch.DoubleTensor
            torch.set_default_tensor_type(torch.DoubleTensor)

        # Set Directories
        self.data_path = params['data_path']
        self.save_path = params['save_path']
        self.model_dir = params['model_dir']
        self.fig_dir = params['fig_dir']
        self.results_dir = params['results_dir']

        # Create Model Name
        self.model_name = self.createModelName()
        print("\n## Model name: \n{:}".format(self.model_name))
        self.save_model_path = self.getModelDir() + "/model"
    
        self.makeDirectories()

        self.display_output = params["display_output"]
        self.num_test_ICS = params["num_test_ICS"]
        self.sequence_length = params['sequence_length']
        self.iterative_prediction_length = params["iterative_prediction_length"]
        self.hidden_state_propagation_length = params["hidden_state_propagation_length"]
        self.teacher_forcing_forecasting=params["teacher_forcing_forecasting"]
        self.iterative_state_forecasting=params["iterative_state_forecasting"]
        self.input_dim = params['input_dim']
        self.output_dim = self.input_dim
        self.N_used = params["N_used"]

        self.scaler_tt = params["scaler"]
        self.scaler = scaler(self.scaler_tt)

        self.retrain =  params['retrain']
        self.train_val_ratio = params['train_val_ratio']
        self.batch_size =  params['batch_size']
        self.overfitting_patience =  params['overfitting_patience']
        self.max_epochs =  params['max_epochs']
        self.max_rounds = params['max_rounds']
        self.learning_rate =  params['learning_rate']
        self.weight_decay =  params['weight_decay']

        # fix the random seed
        self.random_seed = params['random_seed']
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        if self.gpu:
            torch.cuda.manual_seed(self.random_seed)

        # training sequence
        self.data_sequence = self.getTrainingData()

        # set optimizer
        self.optimizer_str = params["optimizer_str"]

        # Build the network
        self.hidden_layers = [self.params["hidden_layer_size"]] * self.params["num_hidden_layers"]
        input_dim = self.input_dim

        # self.net_Dict = nn.ModuleDict({'GRU': nn.GRU(input_dim, self.hidden_layers[-1]),
        #                                'MLP': nn.Linear(self.hidden_layers[-1], self.rnn_state_dim, bias=True),
        #                                })
        # self.net = nn.Sequential(*self.net_Dict.values())

        self.RNN = nn.ModuleList()
        for layer in range(self.params["num_hidden_layers"]):
            self.RNN.append(nn.GRUCell(input_dim, self.params["hidden_layer_size"]))
            input_dim = self.hidden_layers[layer]
        
        # self.RNN = nn.GRU(input_dim, self.params["hidden_layer_size"], self.params["num_hidden_layers"])
        
        # Output MLP
        self.Output = nn.Linear(self.params["hidden_layer_size"], self.output_dim, bias=True)

        self.net = nn.Sequential(self.RNN, self.Output)

        # self.module_list = [self.RNN, self.Output]

        self.model_parameters = list()
        self.n_trainable_parameters = 0
        self.n_model_parameters = 0
        for layer in self.net:
            self.n_trainable_parameters+=sum(p.numel() for p in layer.parameters() if p.requires_grad)
            self.n_model_parameters+=sum(p.numel() for p in layer.parameters())
            self.model_parameters += layer.parameters()
        
        # Print parameter information:
        print("# Trainable params {:}/{:}".format(self.n_trainable_parameters, self.n_model_parameters))

        # Initialize model parameters
        self.initializeWeights()

        if self.gpu:
            print("\n USING CUDA -> SENDING THE MODEL TO THE GPU.\n")
            self.sendModelToCuda()
            if self.device_count > 1:
                print("# Using {:} GPUs! -> PARALLEL Distributing the batch.".format(self.device_count))
                device_ids = list(range(self.device_count))
                print("# device_ids = {:}".format(device_ids))
                self.net = torch.nn.DataParallel(self.net, device_ids=device_ids)

        self.printGPUInfo()

        # Saving some info file for the model
        info = {"model": self, 
                "params": params, 
                "selfParams": self.params, 
                "name": self.model_name,
                'data_sequence': self.data_sequence,
                }
        info_path = self.getModelDir() + "/info"
        saveData(info, info_path, "pickle")


    def makeDirectories(self):
        os.makedirs(self.getModelDir(), exist_ok=True)
        os.makedirs(self.getFigureDir(), exist_ok=True)
        os.makedirs(self.getResultsDir(), exist_ok=True)

    def getModelDir(self):
        model_dir = self.model_dir + self.model_name
        return model_dir

    def getFigureDir(self):
        fig_dir = self.fig_dir + self.model_name
        return fig_dir

    def getResultsDir(self):
        results_dir = self.results_dir + self.model_name
        return results_dir

    def printGPUInfo(self):
        print("\nCUDA Device available? {:}".format(torch.cuda.is_available()))
        print("Number of devices: {:}".format(self.device_count))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Using device:', device)
        
        if device.type == 'cuda':
            for i in range(self.device_count):
                print("DEVICE NAME {:}: {:}".format(i, torch.cuda.get_device_name(i)))
            print('MEMORY USAGE:')
            print('Memory allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
            print('Max memory allocated:', round(torch.cuda.max_memory_allocated(0)/1024**3,1), 'GB')
            print('Memory cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
            print('MAX memory cached:   ', round(torch.cuda.max_memory_reserved(0)/1024**3,1), 'GB')
            print('\n')

    def getKeysInModelName(self):
        keys = {
            'N_used':'-N_used_', 
            'scaler':'-scaler_',
            'optimizer_str':'-OPT_', 
            'learning_rate':'-LR_',
            'weight_decay':'-L2_',
            'num_hidden_layers':'-GRU_',
            'hidden_layer_size':'x',
            'sequence_length':'-SL_',
        }
        return keys
    
    def createModelName(self):
        keys = self.getKeysInModelName()
        str_ = "GPU-" * self.gpu + self.rnnName
        for key in keys:
            str_ += keys[key] + "{:}".format(self.params[key])
        return str_
    
    def ifAnyIn(self, list_, name):
        for element in list_:
            if element in name:
                return True
        return False
    
    def initializeWeights(self):
        for name, param in self.net.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'weight' in name:
                torch.nn.init.xavier_uniform_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
            else:
                raise ValueError("Name {:} Not Found!".format(name))
        
        return 0

    def setupOptimizer(self, learning_rate):
        params = self.model_parameters

        weight_decay = 0.0
        if self.weight_decay > 0: 
            print("No weight decay in RNN training.")
        print("LEARNING RATE: {:}, WEIGHT DECAY: {:}".format(learning_rate, self.weight_decay))

        if self.optimizer_str == "adam":
            self.optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
        elif self.optimizer_str == "sgd":
            self.optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
        elif self.optimizer_str == "rmsprop":
            self.optimizer = torch.optim.RMSprop(params, lr=learning_rate, weight_decay=weight_decay)
        else:
            raise ValueError("Optimizer {:} not recognized.".format(self.optimizer_str))
        
    def getTrainingData(self):
        data = loadData(self.data_path, "pickle")
        self.dt = data['dt']
        input_sequence = data['train_sequence']
        if len(input_sequence.shape)==1:
            input_sequence = input_sequence.reshape(-1, 1)
        
        N_train, first_dim = input_sequence.shape

        if self.N_used > N_train: 
            print("N_used = {:} > N_train = {:}, Not enough samples in the training data.".format(self.N_used, N_train))
            print("Reset N_used to {:}".format(N_train))
            self.N_used = N_train
        
        input_sequence = input_sequence[:self.N_used, :self.input_dim]
        print("##Using {:}/{:} dimensions and {:}/{:} samples ##".format(self.input_dim, first_dim, self.N_used, N_train))
        
        del data

        return input_sequence
    
    def getStartingPoint(self, input_sequence):
        N = input_sequence.shape[0]
        if N-self.hidden_state_propagation_length-self.sequence_length<=0:
            raise ValueError("The hidden_state_propagation_length is too big. Reduce it. N_data !> H + SL, {:} !> {:} + {:} = {:}".format(N, 
                                                                                                                                          self.hidden_state_propagation_length, 
                                                                                                                                          self.sequence_length, 
                                                                                                                                          self.sequence_length+self.hidden_state_propagation_length)
                                                                                                                                          )
        idx = set(np.arange(self.sequence_length, N - self.hidden_state_propagation_length))
        n_samples = len(idx)
        
        return idx, n_samples

        
    def getBatch(self, sequence, batch_idx):
        input_batch = []
        target_batch = []
        for pred_idx in batch_idx:
            input = sequence[pred_idx-self.sequence_length:pred_idx]
            target = sequence[pred_idx-self.sequence_length+1:pred_idx+1]
            input_batch.append(input)
            target_batch.append(target)
            
        input_batch = self.torch_dtype(np.array(input_batch)).transpose(0,1)  # (sequence_length, batch_size, input_dim)
        target_batch = self.torch_dtype(np.array(target_batch)).transpose(0,1)  # (sequence_length, batch_size, input_dim)

        return input_batch, target_batch
    

    def trainBatch(self, idx_train_epoch, batch_idx, input_sequence, is_train=False):
        idx_train_epoch -= set(batch_idx)
        times_to_remove = np.array([range(j-self.sequence_length, j) for j in batch_idx]).flatten()
        idx_train_epoch -= set(times_to_remove)
        times_to_remove = np.array([range(j, j+self.hidden_state_propagation_length) for j in batch_idx]).flatten()
        idx_train_epoch -= set(times_to_remove)

        # set initial hidden state,  (num_hidden_layer, batch_size, hidden_layer_size)
        initial_hidden_state = Variable(torch.zeros(len(self.hidden_layers), len(batch_idx), self.hidden_layers[-1]))

        losses_vec = []
        num_propagations = int(self.hidden_state_propagation_length//self.sequence_length)

        for _ in range(num_propagations):
            self.optimizer.zero_grad()
            # Get the batch,  (batch_size, sequence_length, input_dim)
            input_batch, target_batch = self.getBatch(input_sequence, batch_idx)
            if self.gpu:
                input_batch = input_batch.cuda()
                target_batch = target_batch.cuda()
                initial_hidden_state = initial_hidden_state.cuda()

            # forward the network
            # output_batch,  (sequence_length, batch_size, output_dim)
            # last_hidden_state,  (num_hidden_layer, batch_size, hidden_layer_size)
            # rnn_outputs,  (sequence_length, batch_size, self.hidden_layers[-1])
            output_batch, last_hidden_state = self.forward(input_batch, # (sequence_length, batch_size, input_dim)
                                                           initial_hidden_state,
                                                           is_train=is_train,
                                                           )
            
            # loss
            loss_fwd = (output_batch - target_batch).pow(2.0).mean(2).mean(1).mean()

            if is_train:
                loss_fwd.backward()
                self.optimizer.step()
            
            loss_fwd = loss_fwd.cpu().detach().numpy()
            losses_batch = np.array([loss_fwd])

            last_hidden_state = last_hidden_state.detach()

            losses_vec.append(losses_batch)

            # the initial hidden state computed from the previously processed fragment
            initial_hidden_state = last_hidden_state

            # Update batch index
            batch_idx = np.array(batch_idx) + self.sequence_length

        losses = [np.mean(np.array(losses_vec))]

        return idx_train_epoch, losses
    

    def trainEpoch(self, idx, input_sequence, is_train=False):
        idx_epoch = idx.copy()
        epoch_losses_vec = []
        stop_limit = np.max([self.hidden_state_propagation_length, self.batch_size])
        stop_limit = np.min([stop_limit, len(idx)])
        while len(idx_epoch) >= stop_limit:
            batch_idx = random.sample(idx_epoch, self.batch_size)
            idx_epoch, losses = self.trainBatch(idx_epoch,
                                                batch_idx,
                                                input_sequence,
                                                is_train=is_train,
                                                )
            epoch_losses_vec.append(losses)
        epoch_losses = np.mean(np.array(epoch_losses_vec), axis=0)

        return epoch_losses
    
    def trainRound(self, idx_train, idx_val, train_sequence, val_sequence):
        hitWallTime = False

        # Setting the initial learning rate
        if self.rounds_iter==0:
            self.learning_rate_round = self.learning_rate
            self.previous_round_converged = 0
        elif self.previous_round_converged == 0:
            self.learning_rate_round = self.learning_rate_round
            self.previous_round_converged = 0
        elif self.previous_round_converged == 1:
            self.previous_round_converged = 0
            self.learning_rate_round = 0.5 * self.learning_rate_round

        self.setupOptimizer(self.learning_rate_round)

        if self.retrain == 1:
            print("Restoring Model")
            self.loadModel()
        else:
            # saving the initial model
            print("Saving the Initial Model")
            torch.save(self.net.state_dict(), self.save_model_path)
            
        print("##### Round: {:}, Learning Rate={:} #####".format(self.rounds_iter, self.learning_rate_round))

        losses_train = self.trainEpoch(idx_train, train_sequence, is_train=False)
        losses_val = self.trainEpoch(idx_val, val_sequence, is_train=False)

        print("Initial (New Round):  EP{:} - R{:}".format(self.epochs_iter, self.rounds_iter))

        self.min_val_total_loss = losses_val[0]
        self.loss_total_train = losses_train[0]

        rnn_loss_round_train_vec = []
        rnn_loss_round_val_vec = []

        rnn_loss_round_train_vec.append(losses_train[0])
        rnn_loss_round_val_vec.append(losses_val[0])

        self.loss_total_train_vec.append(losses_train[0])
        self.loss_total_val_vec.append(losses_val[0])

        for epochs_iter in range(self.epochs_iter, self.max_epochs+1):
            epoch_time_start = time.time()
            epochs_in_round = epochs_iter - self.epochs_iter
            self.epochs_iter_global = epochs_iter

            losses_train = self.trainEpoch(idx_train, train_sequence, is_train=True)
            losses_val = self.trainEpoch(idx_val, val_sequence, is_train=False)

            rnn_loss_round_train_vec.append(losses_train[0])
            rnn_loss_round_val_vec.append(losses_val[0])
            self.loss_total_train_vec.append(losses_train[0])
            self.loss_total_val_vec.append(losses_val[0])

            # Print status to screen
            print("##########################################################################################")
            epoch_duration = time.time() - epoch_time_start
            status = "EP={:} - R={:} - ER={:} - [ TIME= {:} ] - LR={:1.2E}".format(epochs_iter, 
                                                                                   self.rounds_iter, 
                                                                                   epochs_in_round, 
                                                                                   secondsToTimeStr(epoch_duration),
                                                                                   self.learning_rate_round
                                                                                   )
            print(status)
            print("# {:s}-losses: ".format('TRAIN')+"= {:1.2E} |".format(losses_train[0]))
            print("# {:s}-losses: ".format('VAL')+"= {:1.2E} |".format(losses_val[0]))

            if losses_val[0] < self.min_val_total_loss:
                print(" Saving Model")
                self.min_val_total_loss = losses_val[0]
                self.loss_total_train = losses_train[0]
                torch.save(self.net.state_dict(), self.save_model_path)
            
            if epochs_in_round > self.overfitting_patience:
                if all(self.min_val_total_loss < rnn_loss_round_val_vec[-self.overfitting_patience:]):
                    self.previous_round_converged = True
                    break
            
            # # LEARNING RATE SCHEDULER (PLATEU ON VALIDATION LOSS)
            # if self.optimizer_str == "adam": self.scheduler.step(losses_val[0])
            self.tqdm.update(1)
            hitWallTime = self.hitWallTime()
            if hitWallTime:
                break
        
        self.rounds_iter += 1
        self.epochs_iter = epochs_iter
        return hitWallTime

    def hitWallTime(self):
        training_time = time.time() - self.start_time
        if training_time > self.train_time_limit:
            print('\n Maximum training time reached. Saving the model now...')
            self.tqdm.close()
            self.saveModel()
            return True
        else:
            return False

    def train(self):

        # input_sequence = self.getTrainingData()
        input_sequence = self.scaler.scaleData(self.data_sequence)

        train_sequence, val_sequence = divideData(input_sequence, self.train_val_ratio)
        idx_train, n_train_samples = self.getStartingPoint(train_sequence)
        idx_val, n_val_samples = self.getStartingPoint(val_sequence)

        print("\n NUMBER OF TRAINING SAMPLES: {:d}".format(n_train_samples))
        print("\n NUMBER OF VALIDATION SAMPLES: {:d}".format(n_val_samples))
        print("\n Start training \n")

        self.loss_total_train_vec = []
        self.loss_total_val_vec = []
        hitWallTime = False

        self.epochs_iter = 0
        self.epochs_iter_global = self.epochs_iter
        self.rounds_iter = 0

        self.tqdm = tqdm(total=self.max_epochs)
        while self.epochs_iter < self.max_epochs and self.rounds_iter < self.max_rounds:
            hitWallTime = self.trainRound(idx_train=idx_train,
                                          idx_val=idx_val,
                                          train_sequence=train_sequence,
                                          val_sequence=val_sequence,
                                          )
            if hitWallTime:
                break
        
        if not hitWallTime:
            if self.epochs_iter == self.max_epochs:
                print("\n## Training finished. Maximum number of epochs reached.\n")
            elif self.rounds_iter == self.max_rounds:
                print("\n## Training finished. Maximum number of rounds reached.\n")
            else:
                print(self.rounds_iter)
                print(self.epochs_iter)
                raise ValueError("## Training finished with unknown error")
            self.saveModel()
            
            PlotTrainingLosses(self, self.loss_total_train_vec, self.loss_total_val_vec, self.min_val_total_loss)



    def forward(self, input, init_hidden_state, is_train=True, is_iterative_forecasting=False, horizon=None, teacher_forcing_forecasting=0):
        # input : (sequence_length, batch_size, input_dim)
        # init_hidden_state : (num_hidden_layer, batch_size, hidden_layer_size)
        if is_train:
            for layer in self.net:
                layer.train()
        else:
            for layer in self.net:
                layer.eval()

        if is_iterative_forecasting:
            # # transpose from batch first to layer first
            # init_hidden_state = init_hidden_state.transpose(0, 1)  # (num_hidden_layer, 1, hidden_layer_size)

            assert len(input.size())==2, f"Dimension of input should be 2, but got {len(input.size())}" 
            assert len(init_hidden_state.size())==2, f"Dimension of init_hidden_state should be 2, but got {len(init_hidden_state.size())}" 

            with torch.set_grad_enabled(is_train):
                outputs = []
                rnn_outputs = []
                T, D = input.size()
                if horizon is None:
                    horizon = T                  
                # When T>1, only input[0,:] is taken into account. 
                # The network propagates its own prediction.
                input_t = input   # (1, input_dim)

                # Iterative Prediction
                for t in range(horizon):
                    # init_hidden_state : (num_hidden_layer, hidden_dim)
                    # rnn_output : (1, hidden_dim)
                    # hidden_state : (num_hidden_layer, hidden_dim)
                    output, _,  next_hidden_state = self.forwardRNN(input_t, init_hidden_state)

                    if t >= teacher_forcing_forecasting:
                        # Iterative prediction:
                        assert teacher_forcing_forecasting != self.sequence_length
                        input_t = output
                    else:
                        input_t = input[t,:]  # (1, input_dim)

                    outputs.append(output[0,:])
                    init_hidden_state = next_hidden_state
                
                outputs = torch.stack(outputs)     # (horizon, output_dim)

            return outputs, next_hidden_state
   
        else:
            
            with torch.set_grad_enabled(is_train):
                # input : (sequence_length, batch_size, input_dim)
                # rnn_outputs : (sequence_length, batch_size, hidden_dim)
                # hidden_states : (num_hidden_layer, batch_size, hidden_dim)
                outputs, rnn_outputs, next_hidden_state = self.forwardRNN(input, init_hidden_state)
                
            return outputs,  next_hidden_state
        
    def forwardRNN(self, input, init_hidden_state):
        T = input.size()[0]
        rnn_outputs = []
        for t in range(T):
            input_t = input[t]  # (batch_size, input_dim)
            next_hidden_state = []
            for layer in range(self.params['num_hidden_layers']):
                layer_output = self.RNN[layer].forward(input_t, init_hidden_state[layer,:])  # (batch_size, hidden_layer_size)
                next_hidden_state.append(layer_output)
                input_t = layer_output

            # update the hidden states
            init_hidden_state = torch.stack(next_hidden_state)  # (num_hidden_layer, batch_size, hidden_layer_size)
            rnn_outputs.append(layer_output)
        
        rnn_outputs = torch.stack(rnn_outputs)   # (T, batch_size, hidden_layer_size)
        next_hidden_state = torch.stack(next_hidden_state)  # (num_hidden_layer, batch_size, hidden_layer_size)
        # Output sequence
        outputs = self.Output(rnn_outputs)  # (T, batch_size, output_dim)

        return outputs, rnn_outputs, next_hidden_state

    def sendModelToCPU(self):
        print("Sending Model to CPU")
        self.net.cpu()    
        return 0

    def sendModelToCuda(self):
        print("Sending Model to CUDA")
        self.net.cuda()               
        return 0
    
    def getModel(self):
        if (not self.gpu) or (self.device_count <=1):
            return self.net
        elif self.gpu and self.device_count>1:
            return self.net.module
        
    def saveModel(self):
        self.total_training_time = time.time() - self.start_time
        if hasattr(self, 'loss_total_train_vec'):
            if len(self.loss_total_train_vec)!=0:
                self.training_time = self.total_training_time / len(self.loss_total_train_vec)
            else:
                self.training_time = self.total_training_time
        else:
            self.training_time = self.total_training_time

        print("Total training time per epoch is {:}".format(secondsToTimeStr(self.training_time)))
        print("Total training time is {:}".format(secondsToTimeStr(self.total_training_time)))

        print("Memory Tracking in MB...")
        process = psutil.Process(os.getpid())
        memory = process.memory_info().rss/1024/1024
        self.memory = memory
        print("Script used {:} MB".format(self.memory))

        data = {
            "params": self.params,
            "model_name": self.model_name,
            "total_training_time": self.total_training_time,
            "training_time": self.training_time,
            "n_trainable_parameters": self.n_trainable_parameters,
            "n_model_parameters": self.n_model_parameters,
            "rnn_loss_train_vec": self.loss_total_train_vec,
            "rnn_loss_val_vec": self.loss_total_val_vec,
            "rnn_min_val_error": self.min_val_total_loss,
            "scaler": self.scaler,
        }

        data_path = self.getModelDir() + "/data"
        saveData(data, data_path, "pickle")

    def loadModel(self, in_cpu=False):          
        if not in_cpu and self.gpu:
            print("# Loading Model in GPU.")
            self.getModel().load_state_dict(torch.load(self.save_model_path))
        else:
            print("# Loading Model in CPU...")
            self.getModel().load_state_dict(torch.load(self.save_model_path, map_location=torch.device('cpu')))
                
        data_path = self.getModelDir() + "/data"
        data = loadData(data_path, "pickle")
        self.scaler = data["scaler"]
        # 
        self.loss_total_train_vec = data["rnn_loss_train_vec"]
        self.loss_total_val_vec = data["rnn_loss_val_vec"]
        self.min_val_total_loss = data["rnn_min_val_error"]
        del data
        return 0
    

    def test(self):
        if self.loadModel()==0:
            test_modes = []
            if self.iterative_state_forecasting:
                test_modes.append('iterative_state_forecasting')
            if self.teacher_forcing_forecasting:
                test_modes.append('teacher_forcing_forecasting')
            
            data = loadData(self.data_path, "pickle")
            input_sequence = data['test_sequence']
            if len(input_sequence.shape)==1:
                input_sequence = input_sequence.reshape(-1,1)
            input_sequence = input_sequence[:, :self.input_dim]
            input_sequence = self.scaler.scaleData(input_sequence, reuse=1)

            n_warmup = int(self.hidden_state_propagation_length//4)

            ic_indices = set(np.arange(self.iterative_prediction_length, input_sequence.shape[0]-self.iterative_prediction_length))

            with torch.no_grad():
                for mode in test_modes:
                    results_path = self.getResultsDir() + "/results_{:}".format(mode)

                    predictions_all = []
                    targets_all = []
                    predictions_augmented_all = []
                    targets_augmented_all = []
                    rmse_all = []
                    rmnse_all = []
                    mnad_all = []
                    num_accurate_pred_005_all = []
                    num_accurate_pred_050_all = []

                    ic_idx = random.sample(ic_indices, self.num_test_ICS)

                    for ic_ in range(self.num_test_ICS):
                        
                        input_sequence_ic = input_sequence[ic_idx[ic_]-n_warmup:ic_idx[ic_]+self.iterative_prediction_length]
                        # prepare the initial hidden states
                        initial_hidden_state = Variable(torch.zeros(len(self.hidden_layers), self.hidden_layers[-1]))
                        # warm-up
                        warmup_data_input = input_sequence_ic[:n_warmup-1]
                        warmup_data_input = self.torch_dtype(warmup_data_input)
                        warmup_data_target = input_sequence_ic[1:n_warmup]

                        if self.gpu:
                            warmup_data_input = warmup_data_input.cuda()
                            initial_hidden_state = initial_hidden_state.cuda()
                        
                        target = input_sequence_ic[n_warmup:n_warmup+self.iterative_prediction_length]
                        target = np.array(target)

                        # forward the network
                        # warmup_data_output : (n_warmup, input_dim)
                        # last_hidden_state : (num_hidden_layer, hidden_layer_size)
                        warmup_data_output, last_hidden_state = self.forward(warmup_data_input,  
                                                                             initial_hidden_state, 
                                                                             is_train=False,
                                                                             )

                        prediction = []
                        if 'iterative' in mode:
                            input_t = input_sequence_ic[n_warmup-1]
                            input_t = input_t[np.newaxis, :]  # (1, input_dim)
                        elif 'teacher' in mode:
                            # (iterative_prediction_length, input_dim)
                            input_t = input_sequence_ic[n_warmup-1:-1]
                        input_t = self.torch_dtype(input_t)
                        if self.gpu:
                            input_t = input_t.cuda()
                            last_hidden_state = last_hidden_state.cuda()

                        time_start = time.time()
                        if 'iterative' in mode:
                            prediction, last_hidden_state = self.forward(input_t,
                                                                         last_hidden_state,
                                                                         is_iterative_forecasting=True,
                                                                         horizon=self.iterative_prediction_length,
                                                                         is_train=False)
                        elif 'teacher' in mode:
                            prediction, last_hidden_state = self.forward(input_t,
                                                                         last_hidden_state,
                                                                         is_iterative_forecasting=False,
                                                                         horizon=self.iterative_prediction_length,
                                                                         is_train=False)
                        
                        time_total = time.time() - time_start
                        time_total_per_iter = time_total / self.iterative_prediction_length

                        prediction = prediction.cpu().detach().numpy()
                        prediction = np.array(prediction)

                        target_augment = np.concatenate((warmup_data_target, target), axis=0)
                        warmup_data_output = warmup_data_output.cpu().detach().numpy()
                        prediction_augment = np.concatenate((warmup_data_output, prediction), axis=0)

                        # descale the data
                        prediction = self.scaler.descaleData(prediction)
                        target = self.scaler.descaleData(target)
                        prediction_augment = self.scaler.descaleData(prediction_augment)
                        target_augment = self.scaler.descaleData(target_augment)
                        # compute errors
                        rmse, rmnse, num_accurate_pred_005, num_accurate_pred_050, mnad = computeErrors(target, 
                                                                                                        prediction, 
                                                                                                        self.scaler.data_std
                                                                                                        )
                        predictions_all.append(prediction)
                        targets_all.append(target)

                        predictions_augmented_all.append(prediction_augment)
                        targets_augmented_all.append(target_augment)

                        rmse_all.append(rmse)
                        rmnse_all.append(rmnse)
                        mnad_all.append(mnad)
                        num_accurate_pred_005_all.append(num_accurate_pred_005)
                        num_accurate_pred_050_all.append(num_accurate_pred_050)


                    predictions_all = np.array(predictions_all)
                    targets_all = np.array(targets_all)
                    predictions_augmented_all = np.array(predictions_augmented_all)
                    targets_augmented_all = np.array(targets_augmented_all)

                    freq_pred, freq_true, sp_true, sp_pred, error_freq = computeFrequencyError(predictions_all, targets_all, self.dt)

                    rmse_all = np.array(rmse_all)
                    rmnse_all = np.array(rmnse_all)
                    mnad_all = np.array(mnad_all)
                    num_accurate_pred_005_all = np.array(num_accurate_pred_005_all)
                    num_accurate_pred_050_all = np.array(num_accurate_pred_050_all)

                    rmnse_avg = np.mean(rmnse_all)
                    rmnse_avg_over_ics = np.mean(rmnse_all, axis=0)
                    mnad_avg = np.mean(mnad_all)
                    mnad_avg_over_ics = np.mean(mnad_all, axis=0)

                    num_accurate_pred_005_avg = np.mean(num_accurate_pred_005_all)
                    num_accurate_pred_050_avg = np.mean(num_accurate_pred_050_all)

                    results = {
                        "dt": self.dt,
                        'lambda_max': np.max(data['Lyapunov_exponents']),
                        "testing_mode": mode,
                        'ic_indices': ic_indices,
                        'time_total_per_iter': time_total_per_iter,
                        'rmse_all': rmse_all,
                        'rmnse_avg': rmnse_avg,
                        'rmnse_all': rmnse_all,
                        'rmnse_avg_over_ics': rmnse_avg_over_ics,
                        'mnad_avg': mnad_avg,
                        'mnad_avg_over_ics': mnad_avg_over_ics,
                        'num_accurate_pred_005_avg': num_accurate_pred_005_avg,
                        'num_accurate_pred_050_avg': num_accurate_pred_050_avg,
                        "predictions_all": predictions_all,
                        "targets_all": targets_all,
                        "predictions_augmented_all": predictions_augmented_all,
                        "targets_augmented_all": targets_augmented_all,
                        "freq_pred": freq_pred,
                        "freq_true": freq_true,
                        "sp_true": sp_true,
                        "sp_pred": sp_pred,
                        "error_freq": error_freq,
                    }

                    if 'delay' in data.keys():
                        results.update({'delay': True, 
                                        'mem_stride': data['mem_stride'],
                                        'embedding_dimension': data['embedding_dimension']})

                    saveData(results, results_path, "pickle")

        return 0

    def PlotResults(self, NLyapTime, plotAttractor=True, plotEmbed=False, plotContour=False):
        if self.loadModel()==0:
            test_modes = []
            if self.iterative_state_forecasting:
                test_modes.append('iterative_state_forecasting')
            if self.teacher_forcing_forecasting:
                test_modes.append('teacher_forcing_forecasting')

            for mode in test_modes:
                results_path = self.getResultsDir() + "/results_{:}".format(mode)
                results = loadData(results_path)

                dt = results['dt']
                lambda_max = results['lambda_max']
                NPlot = int(NLyapTime/lambda_max/dt)
                predictions_augmented = results["predictions_augmented_all"][0]
                targets_augmented = results["targets_augmented_all"][0]

                predictionPlot = predictions_augmented[:NPlot]
                targetPlot = targets_augmented[:NPlot]

                PlotPredictions(self,
                                mode=mode,
                                target=targetPlot,
                                prediction=predictionPlot,
                                dt=dt, 
                                lambda_max=lambda_max,
                                plotEmbed=plotEmbed, 
                                plotContour=plotContour)

                if plotAttractor:
                    # plot the attractor in a plane (the first two components)
                    if 'delay' not in results.keys():
                        fig_path = self.getFigureDir() + "/{:}_{:}_attractor2d.{:}".format(self.rnnChaosName, mode, FIGTYPE)
                        fig = plt.figure(figsize=(13,10))
                        ax = fig.add_subplot()
                        plt.plot(predictions_augmented[:,0], predictions_augmented[:,1], color=color_dict['deepskyblue'], linewidth=1.0, label='prediction')
                        plt.plot(targets_augmented[:,0], targets_augmented[:,1], color=color_dict['tomato'], linewidth=1.0, label='target')
                        ax.axis('off')
                        plt.tight_layout()
                        plt.savefig(fig_path)
                        plt.close()
                    else:
                        predictions_augmented_embed = list()
                        targets_augmented_embed = list()
                        d = results['embedding_dimension']
                        n = predictions_augmented.shape[0]
                        m = results['mem_stride']
                        for i in range(d):
                            predictions_augmented_embed.append(predictions_augmented[i * m : -(d - i) * m, 0])
                            targets_augmented_embed.append(targets_augmented[i * m : -(d - i) * m, 0])

                        predictions_augmented_embed = np.vstack(predictions_augmented_embed)[:, m : (n + m)].T
                        targets_augmented_embed = np.vstack(targets_augmented_embed)[:, m : (n + m)].T

                        fig_path = self.getFigureDir() + "/{:}_{:}_attractor2d.{:}".format(self.rnnChaosName, mode, FIGTYPE)
                        fig = plt.figure(figsize=(13,10))
                        ax = fig.add_subplot()
                        plt.plot(predictions_augmented_embed[:,0], predictions_augmented_embed[:,1], color=color_dict['deepskyblue'], linewidth=1.0, label='prediction')
                        plt.plot(targets_augmented_embed[:,0], targets_augmented_embed[:,1], color=color_dict['tomato'], linewidth=1.0, label='target')
                        ax.axis('off')
                        plt.tight_layout()
                        plt.savefig(fig_path)
                        plt.close()

        return 0
        
    # -----------------------------------------------------------------------------------
    #  The following codes are for calculating the Lyapunov exponents of RNN
    # -----------------------------------------------------------------------------------

    # def SingleLayerJac(self, net, h_t, h_t_1):
    #     partial = []
    #     for i in range(h_t.size()[1]):
    #         temp = torch.zeros(h_t.size())
    #         temp[0][i] = 1.0
    #         h_t.backward(temp, retain_graph=True)
    #         grads = h_t_1.grad.data.cpu().numpy().copy()
    #         partial.append(grads[0].copy())
    #         h_t_1.grad.data.zero_()
    #     Jac = np.array(partial)
    #     net.zero_grad()
    #     return Jac

    def SingleLayerJac(self, next_hidden_states, hidden_states):
        NL = hidden_states.shape[0]
        NS = hidden_states.shape[1]
        Jac = np.zeros((NL*NS, NL*NS))
        for layer in range(NL):
            layer_indx = torch.zeros((1, NL))
            layer_indx[0, layer] = 1.0
            h_t = layer_indx@next_hidden_states
            for i in range(layer, -1, -1):
                partial = []
                for j in range(NS):
                    temp = torch.zeros((1, NS))
                    temp[0, j] = 1.0
                    h_t.backward(temp, retain_graph=True)
                    grads = hidden_states.grad.data.cpu().numpy().copy()
                    partial.append(grads[i].copy())
                    hidden_states.grad.data.zero_()
                Jac[layer*NS:(layer+1)*NS, i*NS:(i+1)*NS] = np.array(partial)
                
        self.net.zero_grad()
        return Jac

    def rnnProp(self, hidden_states, jac_method='analytic'):
        NS = self.params["hidden_layer_size"]
        NL = self.params["num_hidden_layers"]
        ## calculate the Jacobian
        if jac_method == 'auto':
            hidden_states = Variable(hidden_states, requires_grad=True)
            last_indx = torch.zeros((1, NL))
            last_indx[0,-1] = 1.0
            o_t = self.Output(last_indx@hidden_states)
            next_hidden_states = []
            for layer in range(NL):
                if layer == 0:
                    x_in = o_t[0]
                layer_output = self.RNN[layer].forward(x_in, hidden_states[layer])
                next_hidden_states.append(layer_output)
                x_in = layer_output
                
            next_hidden_states = torch.stack(next_hidden_states)
            jac_auto = self.SingleLayerJac(next_hidden_states, hidden_states)

            return next_hidden_states, jac_auto

        elif jac_method == 'analytic':
            Jac_ana = torch.zeros(NS, NS)

            r_idx = slice(0*NS, 1*NS)
            z_idx = slice(1*NS, 2*NS)
            n_idx = slice(2*NS, 3*NS)

            y1 = []
            y2 = []
            next_hidden_states = []
            W_out = self.Output.weight
            b_out = self.Output.bias

            layer = 0
            # extract the network weights
            W_ih = self.RNN[layer].weight_ih   # (W_ir | W_iz | W_in)
            W_hh = self.RNN[layer].weight_hh   # (W_hr | W_hz | W_hn)
            b_ih = self.RNN[layer].bias_ih     # (b_ir | b_iz | b_in)
            b_hh = self.RNN[layer].bias_hh     # (b_hr | b_hz | b_hn)

            o_t = self.Output(hidden_states[-1])
            x_in = o_t
            h_t_1 = hidden_states[layer]
            # y1 = [y1r, y1z, y1n], y2 = [y2r, y2z, y2n]
            # y1.shape = (NL, 3*hidden_layer_size) 
            y1.append(W_ih@x_in + b_ih)  
            y2.append(W_hh@h_t_1 + b_hh)  
            y = y1[layer] + y2[layer] 
            r_t = sig(y[r_idx])
            z_t = sig(y[z_idx])
            n_t = torch.tanh(y1[layer][n_idx] + r_t * y2[layer][n_idx])
            layer_output = z_t*h_t_1 + (1-z_t)*n_t
            next_hidden_states.append(layer_output)
            x_in = layer_output

            b0 = torch.diag_embed(1 - z_t)
            b1 = sech(y1[layer][n_idx]+r_t*y2[layer][n_idx])**2
            b2_h = torch.diag_embed(r_t)@W_hh[n_idx,:]
            b2_x = W_ih[n_idx,:] + torch.diag_embed(y2[layer][n_idx])@sigmoid_p(y[r_idx])@W_ih[r_idx,:]
            b3_h = torch.diag_embed(y2[layer][n_idx])@sigmoid_p(y[r_idx])@W_hh[r_idx,:]

            Jx = torch.diag_embed(h_t_1-n_t)@sigmoid_p(y[z_idx])@W_ih[z_idx,:] + b0@(b1@b2_x)
            Jh = sigmoid(y[z_idx]) + torch.diag_embed(h_t_1-n_t)@sigmoid_p(y[z_idx])@W_hh[z_idx,:] + b0@(b1@(b2_h + b3_h))

            Jac_ana[layer*NS:(layer+1)*NS, layer*NS:(layer+1)*NS] = Jh + Jx@W_out
            next_hidden_states = torch.stack(next_hidden_states)
            # next_hidden_states = next_hidden_states.detach().cpu().numpy()
            Jac_ana = Jac_ana.cpu().detach().numpy().copy()
            return next_hidden_states, Jac_ana

        else:
            raise ValueError("Method for Calculating the Jacobian Not Found!")
    
    def calculate(self,):

        if self.loadModel()==0:

            num_lyaps = self.params["num_lyaps"]

            print("Inside calculation.")
            n_warmup = int(self.hidden_state_propagation_length//4)
            iterative_prediction_length = self.iterative_prediction_length
            print("Number of warm-up steps={:}.".format(n_warmup))

            data = loadData(self.data_path, "pickle")
            input_sequence = data['test_sequence'][:, :self.input_dim]
            del data

            ic_indices = set(np.arange(iterative_prediction_length, input_sequence.shape[0]-iterative_prediction_length))
            ic_idx = random.sample(ic_indices, 1)[0]
            
            input_sequence = input_sequence[ic_idx-n_warmup:ic_idx+self.iterative_prediction_length]
            input_sequence = self.scaler.scaleData(input_sequence, reuse=1)

            warmup_data_input = input_sequence[:n_warmup-1]
            # warmup_data_input = warmup_data_input[:, np.newaxis, :]  # (n_warmup-1, input_dim)
            warmup_data_input = self.torch_dtype(warmup_data_input)

            initial_hidden_state = Variable(torch.zeros(self.params['num_hidden_layers'], 
                                                        self.params['hidden_layer_size']))
            print("Forwarding the warm-up period.")

            _, _, hidden_states = self.forwardRNN(warmup_data_input, initial_hidden_state)

            norm_time = 10
            TT = int(np.floor(iterative_prediction_length/norm_time))
            TT_every = 100
            TT_save = 100
            TRESH = 1e-4

            pl = TT*norm_time
            RDIM = input_sequence.shape[1]

            delta_dim = self.params['hidden_layer_size']

            print("Orthonormal delta of dimensions {:}x{:}.".format(delta_dim, num_lyaps))
            delta = scipy.linalg.orth(np.random.rand(delta_dim, num_lyaps))

            R_ii = np.zeros((num_lyaps, int(pl/norm_time)), dtype=np.complex64)

            LEs_vec = []
            diff_vec = []
            LEs_temp_prev = np.zeros((num_lyaps))
            ITER = 0

            self.tqdm = tqdm(total=pl+2)
            for t in range(pl):
                next_hidden_states, jacobian = self.rnnProp(hidden_states, jac_method='analytic')
                self.net.zero_grad()
                delta = jacobian@delta
                if t % norm_time == 0:
                    QQ, RR = np.linalg.qr(delta)  # QQ (delta_dim, num_lyaps), RR (num_lyaps, num_lyaps)
                    delta = QQ[:, :num_lyaps]
                    R_ii[:,int(t/norm_time)-1] = log(np.diag(RR[:num_lyaps, :num_lyaps]))

                    if (t/norm_time) % TT_every == 0:
                        LEs_temp = np.real(np.sum(R_ii, 1))/((t-1)*self.dt)
                        LEs_temp = np.sort(LEs_temp)
                        diff = np.linalg.norm(LEs_temp - LEs_temp_prev)
                        ITER = ITER + 1
                        LEs_temp_prev = LEs_temp
                        print("Time {:}/{:}, {:3.2f}%".format(t, pl, t/pl*100.0))
                        print("LE: {:}".format(LEs_temp))
                        print("Difference {:.4f}".format(diff))
                        diff_vec.append(diff)
                        LEs_vec.append(LEs_temp)
                        if diff < TRESH:
                            print("TERMINATED AFTER {:} ITERATIONS".format(ITER))
                            break                     
                hidden_states = next_hidden_states.detach()
                self.tqdm.update(1)
            self.tqdm.close()

            LEs = LEs_vec[-1]
            LEs_sum = np.sum(LEs)
            print("Lyapunov Exponents are:")
            print(LEs)
            print("Sum of Lyapunov Exponents is:")
            print(LEs_sum)

            results = {
                        "num_lyaps": num_lyaps,
                        "norm_time": norm_time,
                        "dt": self.dt,
                        "TT": TT,
                        "TT_every": TT_every,
                        "TRESH": TRESH,
                        "ITER": ITER,
                        "diff_vec": diff_vec,
                        "LEs_vec": LEs_vec,
                        "LEs_sum": LEs_sum,
                        "LEs": LEs_vec[-1],
                    }
            data_path = self.getResultsDir() + '/{:}_LE_results_num_lyaps{:}'.format(self.rnnChaosName, num_lyaps)
            saveData(results, data_path, 'pickle')



        
    
    

    





























