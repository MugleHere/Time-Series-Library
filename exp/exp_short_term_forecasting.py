from data_provider.data_factory import data_provider
from data_provider.m4 import M4Meta
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.losses import mape_loss, mase_loss, smape_loss
from utils.m4_summary import M4Summary
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import pandas
import sys
import csv
from tqdm.auto import tqdm  

from tqdm import tqdm

import json
from pathlib import Path

warnings.filterwarnings('ignore')


class Exp_Short_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Short_Term_Forecast, self).__init__(args)

    def _build_model(self):
        if self.args.data == 'm4':
            self.args.pred_len = M4Meta.horizons_map[self.args.seasonal_patterns]  # Up to M4 config
            self.args.seq_len = 2 * self.args.pred_len  # input_len = 2*pred_len
            self.args.label_len = self.args.pred_len
            self.args.frequency_map = M4Meta.frequency_map[self.args.seasonal_patterns]
        model = self.model_dict[self.args.model](self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self, loss_name='MSE'):
        if loss_name == 'MSE':
            return nn.MSELoss()
        elif loss_name == 'MAPE':
            return mape_loss()
        elif loss_name == 'MASE':
            return mase_loss()
        elif loss_name == 'SMAPE':
            return smape_loss()
    @staticmethod
    def _append_jsonl(path: str, row: dict):
        if not path:
            return
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
            f.flush()

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')

        # If exp_dir is set, keep everything inside that folder without adding a long 'setting' subdir
        if getattr(self.args, "exp_dir", None):
            path = self.args.checkpoints
        else:
            path = os.path.join(self.args.checkpoints, setting)

        os.makedirs(path, exist_ok=True)


        # NEW: metrics jsonl location (set by run.py)
        metrics_path = getattr(self.args, "metrics_path", None)

        start_time = time.time()
        best_val = float("inf")
        best_epoch = 0


        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion(self.args.loss)

        log_path = os.path.join(path, "loss_log.csv")
        log_exists = os.path.exists(log_path)

        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            if not log_exists:
                writer.writerow(["epoch", "train_loss", "val_loss", "lr"])

            for epoch in range(self.args.train_epochs):
                self.model.train()
                epoch_time = time.time()

                train_loss = []
                loader_iter = train_loader
                quiet = getattr(self.args, "quiet", False)

                if not quiet:
                    loader_iter = tqdm(
                        train_loader,
                        desc=f"Epoch {epoch+1}/{self.args.train_epochs}",
                        leave=True,
                        file=sys.stdout,
                        dynamic_ncols=True
                    )

                for batch_x, batch_y, batch_x_mark, batch_y_mark in loader_iter:
                    model_optim.zero_grad()

                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)

                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).to(self.device)

                    outputs = self.model(batch_x, None, dec_inp, None)
                    outputs = outputs[:, -self.args.pred_len:, :]
                    true = batch_y[:, -self.args.pred_len:, :]

                    loss = criterion(outputs, true)
                    loss.backward()
                    model_optim.step()

                    loss_item = float(loss.item())
                    train_loss.append(loss_item)
                    if not quiet:
                        loader_iter.set_postfix(loss=loss_item)

                train_loss = float(np.mean(train_loss))
                vali_loss = float(self.vali(vali_data, vali_loader, criterion))
                lr = float(model_optim.param_groups[0]["lr"])
                elapsed = time.time() - start_time

                if vali_loss < best_val:
                    best_val = vali_loss
                    best_epoch = epoch + 1

                self._append_jsonl(metrics_path, {
                    "event": "epoch_end",
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": vali_loss,
                    "best_val": best_val,
                    "best_epoch": best_epoch,
                    "lr": lr,
                    "elapsed_sec": elapsed,
                })


                last_ckpt = {
                    "epoch": epoch + 1,
                    "model": self.model.state_dict(),
                    "optimizer": model_optim.state_dict(),
                    "args": vars(self.args),
                    "train_loss": train_loss,
                    "val_loss": vali_loss,
                }
                torch.save(last_ckpt, os.path.join(path, "last.pth"))

                lr = float(model_optim.param_groups[0]["lr"])

                writer.writerow([epoch + 1, train_loss, vali_loss, lr])
                f.flush()

                #print(
                #    f"Epoch: {epoch+1} | Steps: {train_steps} | "
                #    f"Train Loss: {train_loss:.7f} | Vali Loss: {vali_loss:.7f} | "
                #    f"Time: {time.time() - epoch_time:.1f}s"
                #)
                if not quiet:
                    loader_iter.close()
                    tqdm.write(f"Epoch {epoch+1} | Train Loss: {train_loss:.7f} | Vali Loss: {vali_loss:.7f} | LR: {lr:.6f}")

                early_stopping(vali_loss, self.model, path)
                if early_stopping.early_stop:
                    self._append_jsonl(metrics_path, {
                        "event": "early_stop",
                        "epoch": epoch + 1,
                        "best_val": best_val,
                        "best_epoch": best_epoch,
                        "elapsed_sec": time.time() - start_time,
                    })

                    if not quiet:
                        tqdm.write("Early stopping")
                    break

                adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = os.path.join(path, "checkpoint.pth")
        if os.path.exists(best_model_path):
            self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
        else:
            print(f"[warn] No checkpoint found at {best_model_path}. Returning last epoch weights.")
        return self.model


    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        losses = []
        quiet = getattr(self.args, "quiet", False)
        loader_iter = vali_loader
        if not quiet:
            loader_iter = tqdm(vali_loader, desc="Val", leave=False, file=sys.stdout, dynamic_ncols=True)
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in loader_iter:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).to(self.device)

                outputs = self.model(batch_x, None, dec_inp, None)
                outputs = outputs[:, -self.args.pred_len:, :]
                true = batch_y[:, -self.args.pred_len:, :]

                loss = criterion(outputs, true)
                losses.append(float(loss.item()))

        self.model.train()
        if len(losses) == 0:
            return float("inf")
        return float(np.mean(losses))

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if len(test_data) == 0:
            print("[test] Empty test set (0 windows). Skipping test.")
            return

        quiet = getattr(self.args, "quiet", False)
        metrics_path = getattr(self.args, "metrics_path", None)
        test_start = time.time()

        # Robust checkpoint load BEFORE inference
        if test:
            # Match train() path logic (Option A)
            if getattr(self.args, "exp_dir", None):
                run_dir = self.args.checkpoints
            else:
                run_dir = os.path.join(self.args.checkpoints, setting)

            ckpt_path = os.path.join(run_dir, "checkpoint.pth")
            if not os.path.exists(ckpt_path):
                ckpt_path = os.path.join(run_dir, "last.pth")

            state = torch.load(ckpt_path, map_location=self.device)
            if isinstance(state, dict) and "model" in state:
                self.model.load_state_dict(state["model"])
            else:
                self.model.load_state_dict(state)


        self.model.eval()
        preds, trues = [], []

        loader_iter = test_loader
        if not quiet:
            loader_iter = tqdm(test_loader, desc="Test", leave=False)


        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in loader_iter:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).to(self.device)

                outputs = self.model(batch_x, None, dec_inp, None)
                outputs = outputs[:, -self.args.pred_len:, :]
                true = batch_y[:, -self.args.pred_len:, :]

                preds.append(outputs.detach().cpu().numpy())
                trues.append(true.detach().cpu().numpy())
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)

        errors = preds - trues
        
        denominator = np.abs(preds) + np.abs(trues)
        mask = denominator > 1e-8

        mse = float(np.mean(errors ** 2))
        rmse = float(np.sqrt(mse))
        mae = float(np.mean(np.abs(errors)))
        smape = float(
            np.mean(
                2.0 * np.abs(errors[mask]) / (denominator[mask])
            )
        )





        self._append_jsonl(metrics_path, {
            "event": "test_end",
            "test_mse": mse,
            "test_rmse": rmse,
            "test_mae": mae,
            "test_smape": smape,
            "elapsed_sec": time.time() - test_start,
        })

        if not quiet:
            print("Test MSE:", mse)
            print("Test RMSE:", rmse)
            print("Test MAE:", mae)
            print("Test sMAPE:", smape)

        # Save results (optional; consider moving under exp_dir later)
        exp_dir = getattr(self.args, "exp_dir", None)
        if exp_dir:
            folder_path = os.path.join(exp_dir, "test_results")
        else:
            folder_path = os.path.join("./test_results", setting)

        os.makedirs(folder_path, exist_ok=True)
        np.save(os.path.join(folder_path, 'preds.npy'), preds)
        np.save(os.path.join(folder_path, 'trues.npy'), trues)
