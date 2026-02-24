
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from utils.dtw_metric import dtw, accelerated_dtw
from utils.augmentation import run_augmentation, run_augmentation_single
###
import csv
import json
import sys
from pathlib import Path
from tqdm.auto import tqdm

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model](self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        return torch.optim.Adam(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=getattr(self.args, "weight_decay", 0.0) #ADDED WEIGHT DECAY
        )


    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion
    
    @staticmethod
    def _append_jsonl(path: str, row: dict):
        if not path:
            return
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
            f.flush()

    def _save_last_ckpt(self, path, epoch, model_optim, train_loss, vali_loss):
        last_ckpt = {
            "epoch": epoch,
            "model": self.model.state_dict() if not isinstance(self.model, nn.DataParallel) else self.model.module.state_dict(),
            "optimizer": model_optim.state_dict(),
            "args": vars(self.args),
            "train_loss": float(train_loss),
            "val_loss": float(vali_loss),
        }
        torch.save(last_ckpt, os.path.join(path, "last.pth"))

    def _load_ckpt_flexible(self, ckpt_path):
        state = torch.load(ckpt_path, map_location=self.device)
        if isinstance(state, dict) and "model" in state:
            self.model.load_state_dict(state["model"])
        else:
            self.model.load_state_dict(state)


 

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:]

                pred = outputs.detach()
                true = batch_y.detach()

                loss = criterion(pred, true)

                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    #def train(self, setting):
    #    train_data, train_loader = self._get_data(flag='train')
    #    vali_data, vali_loader = self._get_data(flag='val')
    #    test_data, test_loader = self._get_data(flag='test')
#
    #    path = os.path.join(self.args.checkpoints, setting)
    #    if not os.path.exists(path):
    #        os.makedirs(path)
#
    #    time_now = time.time()
#
    #    train_steps = len(train_loader)
    #    early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
#
    #    model_optim = self._select_optimizer()
    #    criterion = self._select_criterion()
#
    #    if self.args.use_amp:
    #        scaler = torch.cuda.amp.GradScaler()
#
    #    for epoch in range(self.args.train_epochs):
    #        iter_count = 0
    #        train_loss = []
#
    #        self.model.train()
    #        epoch_time = time.time()
    #        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
    #            iter_count += 1
    #            model_optim.zero_grad()
    #            batch_x = batch_x.float().to(self.device)
    #            batch_y = batch_y.float().to(self.device)
    #            batch_x_mark = batch_x_mark.float().to(self.device)
    #            batch_y_mark = batch_y_mark.float().to(self.device)
#
    #            # decoder input
    #            dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
    #            dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
#
    #            # encoder - decoder
    #            if self.args.use_amp:
    #                with torch.cuda.amp.autocast():
    #                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
#
    #                    f_dim = -1 if self.args.features == 'MS' else 0
    #                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
    #                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
    #                    loss = criterion(outputs, batch_y)
    #                    train_loss.append(loss.item())
    #            else:
    #                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
#
    #                f_dim = -1 if self.args.features == 'MS' else 0
    #                outputs = outputs[:, -self.args.pred_len:, f_dim:]
    #                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
    #                loss = criterion(outputs, batch_y)
    #                train_loss.append(loss.item())
#
    #            if (i + 1) % 100 == 0:
    #                print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
    #                speed = (time.time() - time_now) / iter_count
    #                left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
    #                print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
    #                iter_count = 0
    #                time_now = time.time()
#
    #            if self.args.use_amp:
    #                scaler.scale(loss).backward()
    #                scaler.step(model_optim)
    #                scaler.update()
    #            else:
    #                loss.backward()
    #                model_optim.step()
#
    #        print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
    #        train_loss = np.average(train_loss)
    #        vali_loss = self.vali(vali_data, vali_loader, criterion)
    #        test_loss = self.vali(test_data, test_loader, criterion)
#
    #        print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
    #            epoch + 1, train_steps, train_loss, vali_loss, test_loss))
    #        early_stopping(vali_loss, self.model, path)
    #        if early_stopping.early_stop:
    #            print("Early stopping")
    #            break
#
    #        adjust_learning_rate(model_optim, epoch + 1, self.args)
#
    #    best_model_path = path + '/' + 'checkpoint.pth'
    #    self.model.load_state_dict(torch.load(best_model_path))
#
    #    return self.model

    def train(self, setting):
        
        self._append_jsonl(metrics_path, {
            "event": "run_start",
            "model": str(self.args.model),
            "model_id": str(self.args.model_id),
            "data": str(self.args.data),
            "seq_len": int(self.args.seq_len),
            "label_len": int(self.args.label_len),
            "pred_len": int(self.args.pred_len),
            "batch_size": int(self.args.batch_size),
            "learning_rate": float(self.args.learning_rate),
            "weight_decay": float(getattr(self.args, "weight_decay", 0.0)),
            "dropout": float(getattr(self.args, "dropout", 0.0)),
        })

        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')

        # Path logic: if exp_dir is set, keep everything inside args.checkpoints (already inside exp_dir/checkpoints)
        if getattr(self.args, "exp_dir", None):
            path = self.args.checkpoints
        else:
            path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        metrics_path = getattr(self.args, "metrics_path", None)
        quiet = getattr(self.args, "quiet", False)

        start_time = time.time()
        best_val = float("inf")
        best_epoch = 0

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        # CSV log
        log_path = os.path.join(path, "loss_log.csv")
        log_exists = os.path.exists(log_path)
        f_csv = open(log_path, "a", newline="")
        writer = csv.writer(f_csv)
        if not log_exists:
            writer.writerow(["epoch", "train_loss", "val_loss", "lr", "elapsed_sec"])

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            self.model.train()
            epoch_time = time.time()
            train_losses = []

            loader_iter = train_loader
            if not quiet:
                loader_iter = tqdm(
                    train_loader,
                    desc=f"Epoch {epoch+1}/{self.args.train_epochs}",
                    leave=True,
                    file=sys.stdout,
                    dynamic_ncols=True
                )

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(loader_iter):
                if (not quiet) and epoch == 0 and i == 0:
                    print("batch_x shape:", batch_x.shape)
                    print("batch_x_mark shape:", batch_x_mark.shape)




                model_optim.zero_grad(set_to_none=True)

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # forward
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        true = batch_y[:, -self.args.pred_len:, f_dim:]
                        loss = criterion(outputs, true)
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    true = batch_y[:, -self.args.pred_len:, f_dim:]
                    loss = criterion(outputs, true)
                    loss.backward()
                    model_optim.step()

                loss_item = float(loss.detach().item())
                train_losses.append(loss_item)

                if metrics_path and (i % 200 == 0):

                    self._append_jsonl(metrics_path, {
                        "event": "batch_beat",
                        "epoch": epoch + 1,
                        "iter": i,
                        "loss": float(loss_item),
                        "elapsed_sec": float(time.time() - start_time),
                    })


                if not quiet:
                    loader_iter.set_postfix(loss=loss_item)

            # epoch end
            train_loss = float(np.mean(train_losses)) if len(train_losses) else float("inf")
            vali_loss = float(self.vali(vali_data, vali_loader, criterion))

            lr = float(model_optim.param_groups[0]["lr"])
            elapsed = float(time.time() - start_time)

            if vali_loss < best_val:
                best_val = vali_loss
                best_epoch = epoch + 1

            # JSONL metrics
            self._append_jsonl(metrics_path, {
                "event": "epoch_end",
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": vali_loss,
                "best_val": best_val,
                "best_epoch": best_epoch,
                "lr": lr,
                "elapsed_sec": elapsed,
                "pred_len": int(self.args.pred_len),
                "seq_len": int(self.args.seq_len),
                "label_len": int(self.args.label_len),
                "model": str(self.args.model),
                "model_id": str(self.args.model_id),
                "data": str(self.args.data),
            })

            # Save last.pth
            self._save_last_ckpt(path, epoch + 1, model_optim, train_loss, vali_loss)

            # CSV row
            writer.writerow([epoch + 1, train_loss, vali_loss, lr, elapsed])
            f_csv.flush()

            if not quiet:
                tqdm.write(
                    f"Epoch {epoch+1} | Train {train_loss:.7f} | Val {vali_loss:.7f} | "
                    f"LR {lr:.6f} | {time.time() - epoch_time:.1f}s"
                )

            # early stopping saves checkpoint.pth (model.state_dict only)
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                self._append_jsonl(metrics_path, {
                    "event": "early_stop",
                    "epoch": epoch + 1,
                    "best_val": best_val,
                    "best_epoch": best_epoch,
                    "elapsed_sec": float(time.time() - start_time),
                })
                if not quiet:
                    tqdm.write("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        f_csv.close()

        # load best model
        best_model_path = os.path.join(path, 'checkpoint.pth')
        if os.path.exists(best_model_path):
            self._load_ckpt_flexible(best_model_path)
        else:
            print(f"[warn] No checkpoint found at {best_model_path}. Returning last epoch weights.")
        return self.model


    #def test(self, setting, test=0):
    #    test_data, test_loader = self._get_data(flag='test')
    #    if test:
    #        print('loading model')
    #        self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
#
    #    preds = []
    #    trues = []
    #    folder_path = './test_results/' + setting + '/'
    #    if not os.path.exists(folder_path):
    #        os.makedirs(folder_path)
#
    #    self.model.eval()
    #    with torch.no_grad():
    #        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
    #            batch_x = batch_x.float().to(self.device)
    #            batch_y = batch_y.float().to(self.device)
#
    #            batch_x_mark = batch_x_mark.float().to(self.device)
    #            batch_y_mark = batch_y_mark.float().to(self.device)
#
    #            # decoder input
    #            dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
    #            dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
    #            # encoder - decoder
    #            if self.args.use_amp:
    #                with torch.cuda.amp.autocast():
    #                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
    #            else:
    #                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
#
    #            f_dim = -1 if self.args.features == 'MS' else 0
    #            outputs = outputs[:, -self.args.pred_len:, :]
    #            batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
    #            outputs = outputs.detach().cpu().numpy()
    #            batch_y = batch_y.detach().cpu().numpy()
    #            if test_data.scale and self.args.inverse:
    #                shape = batch_y.shape
    #                if outputs.shape[-1] != batch_y.shape[-1]:
    #                    outputs = np.tile(outputs, [1, 1, int(batch_y.shape[-1] / outputs.shape[-1])])
    #                outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
    #                batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)
#
    #            outputs = outputs[:, :, f_dim:]
    #            batch_y = batch_y[:, :, f_dim:]
#
    #            pred = outputs
    #            true = batch_y
#
    #            preds.append(pred)
    #            trues.append(true)
    #            if i % 20 == 0:
    #                input = batch_x.detach().cpu().numpy()
    #                if test_data.scale and self.args.inverse:
    #                    shape = input.shape
    #                    input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
    #                gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
    #                pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
    #                visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))
#
    #    preds = np.concatenate(preds, axis=0)
    #    trues = np.concatenate(trues, axis=0)
    #    print('test shape:', preds.shape, trues.shape)
    #    preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
    #    trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
    #    print('test shape:', preds.shape, trues.shape)
#
    #    # result save
    #    folder_path = './results/' + setting + '/'
    #    if not os.path.exists(folder_path):
    #        os.makedirs(folder_path)
#
    #    # dtw calculation
    #    if self.args.use_dtw:
    #        dtw_list = []
    #        manhattan_distance = lambda x, y: np.abs(x - y)
    #        for i in range(preds.shape[0]):
    #            x = preds[i].reshape(-1, 1)
    #            y = trues[i].reshape(-1, 1)
    #            if i % 100 == 0:
    #                print("calculating dtw iter:", i)
    #            d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
    #            dtw_list.append(d)
    #        dtw = np.array(dtw_list).mean()
    #    else:
    #        dtw = 'Not calculated'
#
    #    mae, mse, rmse, mape, mspe = metric(preds, trues)
    #    print('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
    #    f = open("result_long_term_forecast.txt", 'a')
    #    f.write(setting + "  \n")
    #    f.write('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
    #    f.write('\n')
    #    f.write('\n')
    #    f.close()
#
    #    np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
    #    np.save(folder_path + 'pred.npy', preds)
    #    np.save(folder_path + 'true.npy', trues)
#
    #    return

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        quiet = getattr(self.args, "quiet", False)
        metrics_path = getattr(self.args, "metrics_path", None)
        test_start = time.time()

        # Determine run directory (for loading ckpt and saving results)
        if getattr(self.args, "exp_dir", None):
            run_dir = self.args.checkpoints
        else:
            run_dir = os.path.join(self.args.checkpoints, setting)

        # Load checkpoint if requested
        if test and (not getattr(self.args, "no_ckpt_load", False)):
            ckpt_path = os.path.join(run_dir, "checkpoint.pth")
            if not os.path.exists(ckpt_path):
                ckpt_path = os.path.join(run_dir, "last.pth")

            if os.path.exists(ckpt_path):
                if not quiet:
                    print(f"[test] loading checkpoint: {ckpt_path}")
                self._load_ckpt_flexible(ckpt_path)
            else:
                print(f"[warn] No checkpoint found in {run_dir}. Using current model weights.")

        preds = []
        trues = []

        # Where to save figures and arrays
        if getattr(self.args, "exp_dir", None):
            folder_path = os.path.join(self.args.exp_dir, "test_results")
            results_path = os.path.join(self.args.exp_dir, "results")
        else:
            folder_path = os.path.join("./test_results", setting)
            results_path = os.path.join("./results", setting)

        os.makedirs(folder_path, exist_ok=True)
        os.makedirs(results_path, exist_ok=True)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                # slice horizon
                outputs = outputs[:, -self.args.pred_len:, :]
                true = batch_y[:, -self.args.pred_len:, :]

                outputs_np = outputs.detach().cpu().numpy()
                true_np = true.detach().cpu().numpy()

                # inverse scaling (consistent)
                if getattr(test_data, "scale", False) and self.args.inverse:
                    shape = true_np.shape
                    # if model outputs fewer channels than data, tile (same as your original)
                    if outputs_np.shape[-1] != true_np.shape[-1]:
                        rep = int(true_np.shape[-1] / outputs_np.shape[-1])
                        outputs_np = np.tile(outputs_np, [1, 1, rep])

                    outputs_np = test_data.inverse_transform(outputs_np.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    true_np = test_data.inverse_transform(true_np.reshape(shape[0] * shape[1], -1)).reshape(shape)

                # apply f_dim slicing after inverse, consistent across tasks
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs_np = outputs_np[:, :, f_dim:]
                true_np = true_np[:, :, f_dim:]

                preds.append(outputs_np)
                trues.append(true_np)

                # visualization
                if (i % 20 == 0) and (not quiet):
                    input_np = batch_x.detach().cpu().numpy()
                    if getattr(test_data, "scale", False) and self.args.inverse:
                        in_shape = input_np.shape
                        input_np = test_data.inverse_transform(input_np.reshape(in_shape[0] * in_shape[1], -1)).reshape(in_shape)

                    # plot last channel by default
                    gt = np.concatenate((input_np[0, :, -1], true_np[0, :, -1]), axis=0)
                    pd = np.concatenate((input_np[0, :, -1], outputs_np[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, f"{i}.pdf"))

        preds = np.concatenate(preds, axis=0) if len(preds) else np.empty((0,))
        trues = np.concatenate(trues, axis=0) if len(trues) else np.empty((0,))

        if preds.size == 0:
            print("[test] Empty predictions. Skipping metrics.")
            return

        # reshape consistent with original
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        # -----------------------------
        # Horizon-wise MSE
        # -----------------------------
        # preds/trues shape: (N_samples, pred_len, n_channels)

        errors = preds - trues  # (N, H, C)

        # mean over samples and channels -> one value per horizon step
        horizon_mse = np.mean(errors ** 2, axis=(0, 2))  # shape: (pred_len,)

        # Save as file
        np.save(os.path.join(results_path, "horizon_mse.npy"), horizon_mse)

        # Log to JSONL
        self._append_jsonl(metrics_path, {
            "event": "test_horizon_mse",
            "horizon_mse": horizon_mse.tolist(),
            "pred_len": int(self.args.pred_len),
            "model": str(self.args.model),
            "model_id": str(self.args.model_id),
            "data": str(self.args.data),
        })


        # optional DTW (kept)
        if self.args.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for k in range(preds.shape[0]):
                x = preds[k].reshape(-1, 1)
                y = trues[k].reshape(-1, 1)
                if (k % 100 == 0) and (not quiet):
                    print("calculating dtw iter:", k)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw_val = float(np.mean(dtw_list))
        else:
            dtw_val = None

        mae, mse, rmse, mape, mspe, smape = metric(preds, trues)
        mae, mse, rmse, mape, mspe, smape = float(mae), float(mse), float(rmse), float(mape), float(mspe), float(smape)

        if not quiet:
            print(f"test mse:{mse}, mae:{mae}, rmse:{rmse}, dtw:{dtw_val}, smape:{smape}")

        # write your legacy txt too (optional)

        legacy_txt = os.path.join(results_path, "result_long_term_forecast.txt")
        with open(legacy_txt, "a") as f:
            f.write(setting + "  \n")
            f.write(f"mse:{mse}, mae:{mae}, rmse:{rmse}, dtw:{dtw_val}, smape:{smape}")
            f.write("\n\n")

        # save arrays
        np.save(os.path.join(results_path, "metrics.npy"), np.array([mae, mse, rmse, mape, mspe, smape], dtype=float))
        np.save(os.path.join(results_path, "pred.npy"), preds)
        np.save(os.path.join(results_path, "true.npy"), trues)

        # JSONL metrics
        self._append_jsonl(metrics_path, {
            "event": "test_end",
            "test_mse": mse,
            "test_rmse": rmse,
            "test_mae": mae,
            "test_mape": mape,
            "test_smape":smape,
            "test_mspe": mspe,
            "test_dtw": dtw_val,
            "elapsed_sec": float(time.time() - test_start),
            "pred_len": int(self.args.pred_len),
            "seq_len": int(self.args.seq_len),
            "label_len": int(self.args.label_len),
            "inverse": bool(self.args.inverse),
            "model": str(self.args.model),
            "model_id": str(self.args.model_id),
            "data": str(self.args.data),
        })

        return
