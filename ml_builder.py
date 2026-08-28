# loader pra .npz
from data.npz_sequence_dataset import NPZSequenceDataset
# loader para dataset que é criado na hora do treino
from data.radar_station_memmap_dataset import RadarStationMemmapDataset

import numpy as np
import random as rd
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import time as tm
import os

from model.stconvs2s import STConvS2S_R, STConvS2S_C
from model.baselines import *
from model.ablation import *
 
from tool.train_evaluate import Trainer, Evaluator
from tool.dataset import NetCDFDataset
from tool.loss import (
    RMSELoss,
    MaskedMAELoss,
    MaskedHuberLoss,
    WeightedMaskedMAELoss,
    WeightedMaskedHuberLoss,
)
from tool.utils import Util

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch import optim

class MLBuilder:

    def __init__(self, config, device):
        
        self.config = config
        self.device = device
        self.dataset_type = 'small-dataset' if (self.config.small_dataset) else 'full-dataset'
        self.step = str(config.step)
        self.dataset_name, self.dataset_file = self.__get_dataset_file()
        self.dropout_rate = self.__get_dropout_rate()
        self.filename_prefix = self.dataset_name + '_step' + self.step

    def __parse_years(self):
        years_arg = self.config.years

        if "-" in years_arg:
            start, end = years_arg.split("-")
            return list(range(int(start), int(end) + 1))

        return [int(y.strip()) for y in years_arg.split(",")]
                
    def run_model(self, number):
        self.__define_seed(number)
        validation_split = 0.2
        test_split = 0.2

        if os.path.isdir(self.dataset_file):
            print(f"Usando RadarStationMemmapDataset a partir de: {self.dataset_file}")
            print(f"Fonte dos targets: {self.config.target_source}")

            years = self.__parse_years()

            train_dataset = RadarStationMemmapDataset(
                radar_root=self.dataset_file,
                years=years,
                t_in=5,
                t_out=5,
                stride=int(self.step),
                split="train",
                target_source=self.config.target_source,
            )

            val_dataset = RadarStationMemmapDataset(
                radar_root=self.dataset_file,
                years=years,
                t_in=5,
                t_out=5,
                stride=int(self.step),
                split="val",
                target_source=self.config.target_source,
            )

            test_dataset = RadarStationMemmapDataset(
                radar_root=self.dataset_file,
                years=years,
                t_in=5,
                t_out=5,
                stride=int(self.step),
                split="test",
                target_source=self.config.target_source,
            )

            dataset_kind = "memmap"
    
        # Loading the dataset
        elif str(self.dataset_file).endswith(".npz"):
            print(f"Usando NPZSequenceDataset a partir de: {self.dataset_file}")

            train_dataset = NPZSequenceDataset(self.dataset_file, split="train")
            val_dataset = NPZSequenceDataset(self.dataset_file, split="val")
            test_dataset = NPZSequenceDataset(self.dataset_file, split="test")

            dataset_kind = "npz"
        else:
            ds = xr.open_mfdataset(self.dataset_file)
            if self.config.small_dataset:
                ds = ds[dict(sample=slice(0, 500))]

            train_dataset = NetCDFDataset(
                ds,
                test_split=test_split,
                validation_split=validation_split,
                first_output_channels=self.config.output_channels
            )
            val_dataset = NetCDFDataset(
                ds,
                test_split=test_split,
                validation_split=validation_split,
                is_validation=True,
                first_output_channels=self.config.output_channels
            )
            test_dataset = NetCDFDataset(
                ds,
                test_split=test_split,
                validation_split=validation_split,
                is_test=True,
                first_output_channels=self.config.output_channels
            )
            dataset_kind = "netcdf"

        if self.config.verbose:
            print(f"Train samples: {len(train_dataset)}")
            print(f"Val samples: {len(val_dataset)}")
            print(f"Test samples: {len(test_dataset)}")

            sample = train_dataset[0]

            if len(sample) == 3:
                sample_x, sample_y, sample_m = sample
                print("Sample X:", sample_x.shape)
                print("Sample Y:", sample_y.shape)
                print("Sample M:", sample_m.shape)
            else:
                sample_x, sample_y = sample
                print("Sample X:", sample_x.shape)
                print("Sample Y:", sample_y.shape)

            print(f"Train on {len(train_dataset)} samples, validate on {len(val_dataset)} samples")

        params = {
            'batch_size': self.config.batch,
            'num_workers': self.config.workers,
            'worker_init_fn': self.__init_seed
        }

        train_sampler = None
        train_shuffle = True

        if self.config.balanced_sampler:
            if dataset_kind != "memmap":
                raise ValueError("--balanced-sampler is only supported for memmap radar/station datasets")

            sampler_thresholds = self.__parse_sampler_thresholds()
            sample_weights, class_counts = train_dataset.get_balanced_sample_weights(sampler_thresholds)
            train_sampler = WeightedRandomSampler(
                weights=torch.DoubleTensor(sample_weights),
                num_samples=len(sample_weights),
                replacement=True
            )
            train_shuffle = False
            print(
                "Balanced sampler enabled | "
                f"thresholds={sampler_thresholds} | "
                f"class_counts={class_counts.tolist()}"
            )

        train_loader = DataLoader(dataset=train_dataset, shuffle=train_shuffle, sampler=train_sampler, **params)
        val_loader = DataLoader(dataset=val_dataset, shuffle=False, **params)
        test_loader = DataLoader(dataset=test_dataset, shuffle=False, **params)

        models = {
            'stconvs2s-r': STConvS2S_R,
            'stconvs2s-c': STConvS2S_C,
            'convlstm': STConvLSTM,
            'predrnn': PredRNN,
            'mim': MIM,
            'conv2plus1d': Conv2Plus1D,
            'conv3d': Conv3D,
            'enc-dec3d': Endocer_Decoder3D,
            'ablation-stconvs2s-nocausalconstraint': AblationSTConvS2S_NoCausalConstraint,
            'ablation-stconvs2s-notemporal': AblationSTConvS2S_NoTemporal,
            'ablation-stconvs2s-r-nochannelincrease': AblationSTConvS2S_R_NoChannelIncrease,
            'ablation-stconvs2s-c-nochannelincrease': AblationSTConvS2S_C_NoChannelIncrease,
            'ablation-stconvs2s-r-inverted': AblationSTConvS2S_R_Inverted,
            'ablation-stconvs2s-c-inverted': AblationSTConvS2S_C_Inverted,
            'ablation-stconvs2s-r-notfactorized': AblationSTConvS2S_R_NotFactorized,
            'ablation-stconvs2s-c-notfactorized': AblationSTConvS2S_C_NotFactorized
        }
        if not (self.config.model in models):
            raise ValueError(f'{self.config.model} is not a valid model name. Choose between: {models.keys()}')

        # Creating the model
        model_builder = models[self.config.model]

        if dataset_kind == "memmap":
            sample_x, sample_y, sample_m = train_dataset[0]

            # sample_x: [C, T, H, W]
            C, T, H, W = sample_x.shape
            input_shape = (1, C, T, H, W)

            # sample_y: [C_out, T_out, H, W]
            output_channels = sample_y.shape[0]

            print("Shape enviado ao modelo:", input_shape)
            print("Output channels:", output_channels)

            model = model_builder(
                input_shape,
                self.config.num_layers,
                self.config.hidden_dim,
                self.config.kernel_size,
                self.device,
                self.dropout_rate,
                int(self.step),
                output_channels=output_channels
            )

        elif dataset_kind == "npz":
            raw_shape = train_dataset.X.shape
            _, T, H, W, C = raw_shape
            input_shape = (1, C, T, H, W)

            print("Shape bruto do dataset:", raw_shape)
            print("Shape enviado ao modelo:", input_shape)

            if len(train_dataset.y.shape) == 5:
                output_channels = train_dataset.y.shape[-1]
            elif len(train_dataset.y.shape) == 4:
                output_channels = train_dataset.y.shape[-1]
            else:
                output_channels = None

            model = model_builder(
                input_shape,
                self.config.num_layers,
                self.config.hidden_dim,
                self.config.kernel_size,
                self.device,
                self.dropout_rate,
                int(self.step),
                output_channels=output_channels
            )

        else:
            input_shape = train_dataset.X.shape
            output_shape = train_dataset.y.shape

            model = model_builder(
                input_shape,
                self.config.num_layers,
                self.config.hidden_dim,
                self.config.kernel_size,
                self.device,
                self.dropout_rate,
                int(self.step),
                output_channels=output_shape[1]
            )

        model.to(self.device)

        criterion = self.__get_loss()
        print(f"Loss function: {criterion.__class__.__name__}")
        opt_params = {'lr': 0.001, 'alpha': 0.9, 'eps': 1e-6}
        optimizer = torch.optim.RMSprop(model.parameters(), **opt_params)
        util = Util(self.config.model, self.dataset_type, self.config.version, self.filename_prefix)

        train_info = {'train_time': 0}
        if self.config.pre_trained is None:
            train_info = self.__execute_learning(model, criterion, optimizer, train_loader, val_loader, util)

        eval_info = self.__load_and_evaluate(model, criterion, optimizer, test_loader, train_info['train_time'], util)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {**train_info, **eval_info}

    def __execute_learning(self, model, criterion, optimizer, train_loader, val_loader, util):
        checkpoint_filename = util.get_checkpoint_filename()    
        trainer = Trainer(model, criterion, optimizer, train_loader, val_loader, self.config.epoch, 
                          self.device, util, self.config.verbose, self.config.patience, self.config.no_stop)
    
        start_timestamp = tm.time()
        # Training the model
        train_losses, val_losses = trainer.fit(checkpoint_filename, is_chirps=self.config.chirps)
        end_timestamp = tm.time()
        # Learning curve
        util.save_loss(train_losses, val_losses)
        util.plot([train_losses, val_losses], ['Training', 'Validation'], 'Epochs', 'Loss',
                  'Learning curve - ' + self.config.model.upper(), self.config.plot)

        train_time = end_timestamp - start_timestamp       
        print(f'\nTraining time: {util.to_readable_time(train_time)} [{train_time}]')
               
        return {'dataset': self.dataset_name,
                'dropout_rate': self.dropout_rate,
                'train_time': train_time
                }
                
    
    def __load_and_evaluate(self, model, criterion, optimizer, test_loader, train_time, util):  
        evaluator = Evaluator(model, criterion, optimizer, test_loader, self.device, util, self.step)
        if self.config.pre_trained is not None:
            # Load pre-trained model
            best_epoch, val_loss = evaluator.load_checkpoint(self.config.pre_trained, self.dataset_type, self.config.model)
        else:
            # Load model with minimal loss after training phase
            checkpoint_filename = util.get_checkpoint_filename() 
            best_epoch, val_loss = evaluator.load_checkpoint(checkpoint_filename)
        
        time_per_epochs = 0
        if not(self.config.no_stop): # Earling stopping during training
            time_per_epochs = train_time / (best_epoch + self.config.patience)
            print(f'Training time/epochs: {util.to_readable_time(time_per_epochs)} [{time_per_epochs}]')
        
        test_rmse, test_mae, test_bias = evaluator.eval(is_chirps=self.config.chirps)
        print(f'Test RMSE: {test_rmse:.4f}\nTest MAE: {test_mae:.4f}\nTest Bias: {test_bias:.4f}')
                        
        return {'best_epoch': best_epoch,
                'val_rmse': val_loss,
                'test_rmse': test_rmse,
                'test_mae': test_mae,
                'test_bias': test_bias,
                'train_time_epochs': time_per_epochs
                }
          
    def __define_seed(self, number):      
        if (~self.config.no_seed):
            # define a different seed in every iteration 
            seed = (number * 10) + 1000
            np.random.seed(seed)
            rd.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic=True
            
    def __init_seed(self, number):
        seed = (number * 10) + 1000
        np.random.seed(seed)

    def __get_loss(self):
        loss_name = self.config.loss

        if loss_name == 'masked-mae':
            return MaskedMAELoss()

        if loss_name == 'masked-huber':
            return MaskedHuberLoss(delta=self.config.huber_delta)

        if loss_name == 'weighted-mae':
            return WeightedMaskedMAELoss(class_weights=self.__parse_loss_weights())

        if loss_name == 'weighted-huber':
            return WeightedMaskedHuberLoss(
                delta=self.config.huber_delta,
                class_weights=self.__parse_loss_weights()
            )

        raise ValueError(f'{loss_name} is not a valid loss name')

    def __parse_loss_weights(self):
        try:
            weights = [float(value.strip()) for value in self.config.loss_weights.split(',')]
        except ValueError:
            raise ValueError('--loss-weights must contain numeric comma-separated values')

        if len(weights) != 4:
            raise ValueError('--loss-weights must contain four values: weak,moderate,strong,extreme')

        if any(weight <= 0 for weight in weights):
            raise ValueError('--loss-weights values must be positive')

        return weights

    def __parse_sampler_thresholds(self):
        try:
            thresholds = [float(value.strip()) for value in self.config.sampler_thresholds.split(',')]
        except ValueError:
            raise ValueError('--sampler-thresholds must contain numeric comma-separated values')

        if len(thresholds) != 3:
            raise ValueError('--sampler-thresholds must contain three values: moderate,strong,extreme')

        if any(threshold <= 0 for threshold in thresholds):
            raise ValueError('--sampler-thresholds values must be positive')

        if thresholds != sorted(thresholds):
            raise ValueError('--sampler-thresholds values must be sorted in ascending order')

        return tuple(thresholds)
        
    def __get_dataset_file(self):
        if self.config.dataset_path is not None:
            dataset_file = self.config.dataset_path
            dataset_name = os.path.splitext(os.path.basename(dataset_file))[0]
        elif self.config.chirps:
            dataset_file = 'data/dataset-chirps-1981-2019-seq5-ystep' + self.step + '.nc'
            dataset_name = 'chirps'
        else:
            dataset_file = 'data/dataset-ucar-1979-2015-seq5-ystep' + self.step + '.nc'
            dataset_name = 'cfsr'
        
        return dataset_name, dataset_file
        
    def __get_dropout_rate(self):
        dropout_rates = {
            'predrnn': 0.5,
            'mim': 0.5
        }
        if self.config.model in dropout_rates:
            dropout_rate = dropout_rates[self.config.model] 
        else:
            dropout_rate = 0.

        return dropout_rate
