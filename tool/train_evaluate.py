import os
import numpy as np

import torch
import torch.nn.functional as F


PRECIP_BINS = [
    ("Fraca (<1.25 mm/15min; <5 mm/h equiv.)", 0.0, 1.25),
    ("Moderada (1.25-6.25 mm/15min; 5-25 mm/h equiv.)", 1.25, 6.25),
    ("Forte (6.25-12.5 mm/15min; 25-50 mm/h equiv.)", 6.25, 12.5),
    ("Extrema (>12.5 mm/15min; >50 mm/h equiv.)", 12.5, float("inf")),
]


def create_metric_stats(precip_bins=PRECIP_BINS, step=None):
    stats = {}

    for label, _, _ in precip_bins:
        if step is None:
            stats[label] = {"se": 0.0, "ae": 0.0, "bias": 0.0, "n": 0}
        else:
            stats[label] = [
                {"se": 0.0, "ae": 0.0, "bias": 0.0, "n": 0}
                for _ in range(step)
            ]

    return stats


def update_metric_stats(stats, output_mm15, target_mm15, mask, precip_bins=PRECIP_BINS, horizon=None):
    valid = mask == 1

    for label, low, high in precip_bins:
        if high == float("inf"):
            class_mask = valid & (target_mm15 >= low)
        else:
            class_mask = valid & (target_mm15 >= low) & (target_mm15 < high)

        n = class_mask.sum().item()

        if n == 0:
            continue

        err = output_mm15[class_mask] - target_mm15[class_mask]
        target_stats = stats[label] if horizon is None else stats[label][horizon]
        target_stats["se"] += torch.sum(err ** 2).item()
        target_stats["ae"] += torch.sum(torch.abs(err)).item()
        target_stats["bias"] += torch.sum(err).item()
        target_stats["n"] += n


def metric_rows_from_stats(stats, split="test"):
    rows = []

    for label, value in stats.items():
        horizon_stats = value if isinstance(value, list) else [value]

        for horizon_idx, item in enumerate(horizon_stats):
            n = item["n"]
            row = {
                "split": split,
                "precipitation_bin": label,
                "horizon": horizon_idx + 1 if isinstance(value, list) else "",
                "n": n,
                "rmse": "",
                "mae": "",
                "bias": "",
            }

            if n > 0:
                row["rmse"] = float(np.sqrt(item["se"] / n))
                row["mae"] = float(item["ae"] / n)
                row["bias"] = float(item["bias"] / n)

            rows.append(row)

    return rows


class Trainer:
    
    def __init__(self, model, loss_fn, optimizer, train_loader, val_loader, 
                 epochs, device, util, verbose, patience, no_stop):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.device = device
        self.verbose = verbose
        self.util = util
        self.early_stopping = EarlyStopping(verbose, patience, no_stop)
        
    def fit(self, filename, is_chirps=False):
        train_losses, val_losses = [], []

        for epoch in range(1,self.epochs+1):
            train_loss = self.__train(is_chirps)
            evaluator = Evaluator(self.model, self.loss_fn, self.optimizer, self.val_loader, self.device, self.util)
            val_loss,_,_ = evaluator.eval(is_test=False, is_chirps=is_chirps)
            if (self.verbose):
                print(f'Epoch: {epoch}/{self.epochs} - loss: {train_loss:.4f} - val_loss: {val_loss:.4f}')
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            self.early_stopping(val_loss, self.model, self.optimizer, epoch, filename)
            if (torch.cuda.is_available()):
                torch.cuda.empty_cache()
            if self.early_stopping.isToStop:
                if (self.verbose):
                    print("=> Stopped")
                break

        return train_losses, val_losses

    def __train(self, is_chirps=False):
        self.model.train()
        epoch_loss = 0.0
        mask_land = self.util.get_mask_land().to(self.device)

        for batch_idx, (inputs, target, mask) in enumerate(self.train_loader):
            inputs = inputs.to(self.device)
            target = target.to(self.device)
            mask = mask.to(self.device)
            # get prediction
            output = self.model(inputs)

            if is_chirps:
                output = mask_land * output
            # with mask
            loss = self.loss_fn(output, target, mask)

            # clear previous gradients 
            self.optimizer.zero_grad()
            # compute gradients
            loss.backward()
            # performs updates using calculated gradients
            self.optimizer.step()
            epoch_loss += loss.item()

        return  epoch_loss/len(self.train_loader)
            
            
class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience
    """
    
    def __init__(self, verbose, patience, no_stop):
        self.verbose = verbose
        self.patience = patience
        self.best_loss = float('inf')
        self.counter = 0
        self.isToStop = False
        self.enable_stop = not no_stop
          
    def __call__(self, val_loss, model, optimizer, epoch, filename):
        is_best = bool(val_loss < self.best_loss)
        if (is_best):
            self.best_loss = val_loss
            self.__save_checkpoint(self.best_loss, model, optimizer, epoch, filename)
            self.counter = 0
        elif (self.enable_stop):
            self.counter += 1
            if (self.verbose):
                print(f'=> Early stopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.isToStop = True
    
    def __save_checkpoint(self, loss, model, optimizer, epoch, filename):
        state = {'model_state_dict': model.state_dict(),
                 'optimizer_state_dict': optimizer.state_dict(),
                 'epoch': epoch,
                 'loss': loss}
        torch.save(state, filename)
        if (self.verbose):
            print ('=> Saving a new best') 
        
    
class Evaluator:
        
    def __init__(self, model, loss_fn, optimizer, data_loader, device, util=None, step=0):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.data_loader = data_loader
        self.util = util
        self.step = int(step)
        self.device = device
       
    def eval(self, is_test=True, is_chirps=False):
        self.model.eval()

        cumulative_rmse, cumulative_mae, cumulative_bias = 0.0, 0.0, 0.0
        observation_rmse = [0] * self.step
        observation_mae = [0] * self.step
        observation_bias = [0] * self.step
        loader_size = len(self.data_loader)

        class_stats = create_metric_stats()
        class_horizon_stats = create_metric_stats(step=self.step)

        mask_land = self.util.get_mask_land().to(self.device)

        with torch.no_grad():
            for batch_i, (inputs, target, mask) in enumerate(self.data_loader):
                inputs = inputs.to(self.device)
                target = target.to(self.device)
                mask = mask.to(self.device)

                output = self.model(inputs)

                if is_chirps:
                    output = mask_land * output

                # Debug somente no primeiro batch do teste final
                if is_test and batch_i == 0:
                    print("\n========== DEBUG PRIMEIRO BATCH ==========")
                    print(f"inputs.shape: {tuple(inputs.shape)}")
                    print(f"target.shape: {tuple(target.shape)}")
                    print(f"mask.shape:   {tuple(mask.shape)}")
                    print(f"output.shape: {tuple(output.shape)}")
                    print()
                    print("Ordem: [batch, canal, tempo, altura, largura]")
                    print(f"Batch:            eixo 0, tamanho {inputs.shape[0]}")
                    print(f"Canais de input:  eixo 1, tamanho {inputs.shape[1]}")
                    print(f"Canais do target: eixo 1, tamanho {target.shape[1]}")
                    print(f"Canais do output: eixo 1, tamanho {output.shape[1]}")
                    print(f"Tempo de input:   eixo 2, tamanho {inputs.shape[2]}")
                    print(f"Tempo de saída:   eixo 2, tamanho {target.shape[2]}")
                    print(f"Altura:           eixo 3, tamanho {target.shape[3]}")
                    print(f"Largura:          eixo 4, tamanho {target.shape[4]}")
                    print("===========================================\n")

                # target e output estão em log1p(mm/15min)
                # para métricas, volta para mm/15min
                #output_mm15 = torch.expm1(output)
                # evitar valores negativos
                output_mm15 = torch.clamp(torch.expm1(output), min=0.0)
                target_mm15 = torch.expm1(target)

                if is_test and batch_i == 0:
                    print("\n===== ESTATÍSTICAS POR HORIZONTE =====")

                    for t in range(target_mm15.shape[2]):
                        target_t = target_mm15[:, :, t, :, :]
                        output_t = output_mm15[:, :, t, :, :]
                        mask_t = mask[:, :, t, :, :]

                        valid = mask_t == 1
                        target_valid = target_t[valid]
                        output_valid = output_t[valid]

                        print(f"\nt+{t + 1}:")
                        print(f"  target shape: {tuple(target_t.shape)}")
                        print(f"  output shape: {tuple(output_t.shape)}")
                        print(f"  pontos válidos: {valid.sum().item()}")

                        if valid.any():
                            print(
                                f"  target: "
                                f"min={target_valid.min().item():.4f}, "
                                f"max={target_valid.max().item():.4f}, "
                                f"mean={target_valid.mean().item():.4f}, "
                                f"std={target_valid.std().item():.4f}"
                            )

                            print(
                                f"  output: "
                                f"min={output_valid.min().item():.4f}, "
                                f"max={output_valid.max().item():.4f}, "
                                f"mean={output_valid.mean().item():.4f}, "
                                f"std={output_valid.std().item():.4f}"
                            )

                    print("=======================================\n")

                diff = output_mm15 - target_mm15

                rmse_loss = torch.sqrt(
                    ((diff ** 2) * mask).sum() / (mask.sum() + 1e-8)
                )

                mae_loss = (
                    torch.abs(diff) * mask
                ).sum() / (mask.sum() + 1e-8)

                bias_loss = (
                    diff * mask
                ).sum() / (mask.sum() + 1e-8)

                cumulative_rmse += rmse_loss.item()
                cumulative_mae += mae_loss.item()
                cumulative_bias += bias_loss.item()

                if is_test:
                    update_metric_stats(class_stats, output_mm15, target_mm15, mask)

                    for i in range(self.step):
                        output_observation = output_mm15[:, :, i, :, :]
                        target_observation = target_mm15[:, :, i, :, :]
                        mask_observation = mask[:, :, i, :, :]

                        diff_obs = output_observation - target_observation

                        rmse_loss_obs = torch.sqrt(
                            ((diff_obs ** 2) * mask_observation).sum()
                            / (mask_observation.sum() + 1e-8)
                        )

                        mae_loss_obs = (
                            torch.abs(diff_obs) * mask_observation
                        ).sum() / (mask_observation.sum() + 1e-8)

                        bias_loss_obs = (
                            diff_obs * mask_observation
                        ).sum() / (mask_observation.sum() + 1e-8)

                        observation_rmse[i] += rmse_loss_obs.item()
                        observation_mae[i] += mae_loss_obs.item()
                        observation_bias[i] += bias_loss_obs.item()
                        update_metric_stats(
                            class_horizon_stats,
                            output_observation,
                            target_observation,
                            mask_observation,
                            horizon=i
                        )

            if is_test:
                self.util.save_examples(
                    inputs.detach().cpu(),
                    target_mm15.detach().cpu(),
                    output_mm15.detach().cpu(),
                    self.step
                )

                print('>>>>>>>>> Metric per observation (lat x lon) at each time step (t)')
                print('RMSE (mm/15min)')
                print(*np.divide(observation_rmse, batch_i + 1), sep=",")
                print('MAE (mm/15min)')
                print(*np.divide(observation_mae, batch_i + 1), sep=",")
                print('Bias (mm/15min)')
                print(*np.divide(observation_bias, batch_i + 1), sep=",")
                print('>>>>>>>>')

                print("\n>>>>>>>>> Metric by precipitation intensity")
                for label, stats in class_stats.items():
                    n = stats["n"]

                    if n == 0:
                        print(f"{label}: n=0")
                        continue

                    rmse = np.sqrt(stats["se"] / n)
                    mae = stats["ae"] / n
                    bias = stats["bias"] / n

                    print(
                        f"{label}: "
                        f"n={n}, "
                        f"RMSE={rmse:.4f} mm/15min, "
                        f"MAE={mae:.4f} mm/15min, "
                        f"Bias={bias:.4f} mm/15min"
                    )
                print(">>>>>>>>")

                print("\n>>>>>>>>> Metric by precipitation intensity and horizon")
                class_horizon_rows = metric_rows_from_stats(class_horizon_stats)
                for row in class_horizon_rows:
                    if row["n"] == 0:
                        print(
                            f'{row["precipitation_bin"]} | '
                            f't+{row["horizon"]}: n=0'
                        )
                        continue

                    print(
                        f'{row["precipitation_bin"]} | '
                        f't+{row["horizon"]}: '
                        f'n={row["n"]}, '
                        f'RMSE={row["rmse"]:.4f} mm/15min, '
                        f'MAE={row["mae"]:.4f} mm/15min, '
                        f'Bias={row["bias"]:.4f} mm/15min'
                    )
                print(">>>>>>>>")

                if self.util is not None:
                    self.util.save_metrics(class_horizon_rows, "intensity_horizon")

        return (
            cumulative_rmse / loader_size,
            cumulative_mae / loader_size,
            cumulative_bias / loader_size
        )
        
    def load_checkpoint(self, filename, dataset_type=None, model=None):
        if not(os.path.isabs(filename)):
            filename = os.path.join('output', dataset_type, 'checkpoints', model.lower(), filename)  
        epoch, loss = 0.0, 0.0
        if os.path.isfile(filename):
            checkpoint = torch.load(filename)
            name = os.path.basename(filename)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            loss = checkpoint['loss']
            print(f'=> Loaded checkpoint {name} (best epoch: {epoch}, validation rmse: {loss:.4f})')
        else:
            print(f'=> No checkpoint found at {filename}')

        return epoch, loss
