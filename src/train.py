import math
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.config import DEVICE, Y_LOG_MIN, Y_LOG_MAX

def clamp_pred_log(p): 
    return torch.clamp(p, min=Y_LOG_MIN, max=Y_LOG_MAX)

def y_inverse(y_log): 
    return np.expm1(y_log).astype(np.float32)

class WarmupCosine(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, max_epochs, eta_min=1e-6, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)
        
    def get_lr(self):
        e = self.last_epoch
        base = self.base_lrs
        if e < self.warmup_epochs:
            s = (e + 1) / max(1, self.warmup_epochs)
            return [lr * s for lr in base]
        t = (e - self.warmup_epochs) / max(1, (self.max_epochs - self.warmup_epochs))
        cos = 0.5 * (1 + math.cos(math.pi * t))
        return [self.eta_min + (lr - self.eta_min) * cos for lr in base]

class EMA:
    def __init__(self, model, decay=0.995):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}
        
    def update(self, model):
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if v.is_floating_point(): 
                    self.shadow[k].mul_(self.decay).add_(v, alpha=1.0 - self.decay)
                else: 
                    self.shadow[k].copy_(v)
                    
    def apply(self, model): 
        model.load_state_dict(self.shadow, strict=True)

@torch.no_grad()
def evaluate_model(model, loader):
    model.eval()
    yt_all, yp_all = [], []
    for *inputs, y_log, _ in loader:
        inputs = [torch.nan_to_num(x.to(DEVICE, non_blocking=True)) for x in inputs]
        preds = clamp_pred_log(model(*inputs))
        
        yt_all.append(y_inverse(y_log.numpy().reshape(-1)))
        yp_all.append(y_inverse(preds.cpu().numpy().reshape(-1)))

    yt, yp = np.concatenate(yt_all), np.concatenate(yp_all)
    return mean_absolute_error(yt, yp), mean_squared_error(yt, yp)**0.5, r2_score(yt, yp)

def train_pipeline(model, train_loader, val_loader, test_loader):
    EPOCHS = 120
    PATIENCE = 25
    MIN_DELTA = 1e-4
    AMP = False 
    
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=5e-3)
    sch = WarmupCosine(opt, warmup_epochs=15, max_epochs=EPOCHS) 
    
    scaler = torch.amp.GradScaler("cuda", enabled=AMP)
    ema = EMA(model, decay=0.995)
    loss_fn = nn.HuberLoss(delta=0.25)
    
    best_val_mae = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(EPOCHS):
        model.train()
        epoch_losses = []
        
        for *inputs, y_log, _ in train_loader:
            inputs = [torch.nan_to_num(x.to(DEVICE, non_blocking=True)) for x in inputs]
            y_log = torch.nan_to_num(y_log.to(DEVICE, non_blocking=True))
            
            opt.zero_grad(set_to_none=True)
            
            with torch.amp.autocast("cuda", enabled=AMP):
                p_log = model(*inputs)
                p_log = clamp_pred_log(p_log)
                loss = loss_fn(p_log, y_log)

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            scaler.step(opt)
            scaler.update()
            
            ema.update(model)
            epoch_losses.append(loss.item())

        sch.step()
        
        saved_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        ema.apply(model)
        val_mae, val_rmse, val_r2 = evaluate_model(model, val_loader)
        model.load_state_dict(saved_state) 
        
        current_lr = opt.param_groups[0]['lr']
        print(f"Epoch {epoch:02d} | train_loss={np.mean(epoch_losses):.4f} | val MAE={val_mae:.4f} RMSE={val_rmse:.4f} | lr={current_lr:.2e}")
        
        if val_mae < best_val_mae - MIN_DELTA:
            best_val_mae = val_mae
            best_state = {k: v.cpu().clone() for k, v in ema.shadow.items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"Early stopping at epoch {epoch:02d} (best val MAE={best_val_mae:.4f}).")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        
    test_mae, test_rmse, test_r2 = evaluate_model(model, test_loader)
    
    print(f"\nBEST val MAE={best_val_mae:.4f}")
    print(f"TEST  MAE={test_mae:.4f} RMSE={test_rmse:.4f} R2={test_r2:.4f}\n")
    
    return test_mae, test_rmse, test_r2