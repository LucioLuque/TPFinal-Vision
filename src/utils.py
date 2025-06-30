import random
import torch
import numpy as np
from PIL import Image
import pandas as pd
from models import get_y_tensors, Upsampler, SRCNN, FSRCNN, FSRCA, FSRCA_PS
from metrics import calculate_average_metrics, calculate_psnr_torch, calculate_psnr
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.amp import autocast, GradScaler
import torchvision.transforms as T
import torch.nn.functional as F
import json
import matplotlib.pyplot as plt
from PIL import Image
import os
import cv2
import time

def deterministic(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

def get_set_images(set_number, factor):
    images = []
    file_path = f"../Datasets/Test/Set{set_number}/image_SRF_{factor}/"
    for i in range(1, set_number + 1):
        hr_image = Image.open(f"{file_path}/img_{i:03d}_SRF_{factor}_HR.png")
        lr_image = Image.open(f"{file_path}/img_{i:03d}_SRF_{factor}_LR.png")
        images.append((hr_image, lr_image))
    return images

def get_test_images(set_numbers=[5, 14], factors=[2, 3, 4]):
    test_images = {set_number: {} for set_number in set_numbers}
    for set_number in set_numbers:
        for factor in factors:
            images = get_set_images(set_number, factor)
            lr_images = [img[1] for img in images]
            hr_images = [img[0] for img in images]
            test_images[set_number][factor] = {
                'lr': lr_images,
                'hr': hr_images
            }
    return test_images

# def get_interpolation_results(all_images, set_numbers=[5, 14], factors=[2, 3, 4], methods = ["nearest", "bilinear", "bicubic"]):
#     results = []
#     for set_number in set_numbers:
#         for factor in factors:
#             for method in methods:
                
#                 lr_images = all_images[set_number][factor]['lr']
#                 hr_images = all_images[set_number][factor]['hr']
#                 model = Upsampler(mode=method, scale_factor=factor)
#                 lr_images = get_y_tensors(lr_images)
#                 hr_images = get_y_tensors(hr_images)

#                 unsampled_images = [model(img) for img in lr_images]
#                 unsampled_images = [img[:, :, factor:-factor, factor:-factor].squeeze() for img in unsampled_images]
#                 hr_images = [img[:, :, factor:-factor, factor:-factor].squeeze() for img in hr_images]

#                 psnr, ssim = calculate_average_metrics(hr_images, unsampled_images, data_range=1.0)

#                 new_row = {
#                     "Dataset": f"Set{set_number}",
#                     "Factor": factor,
#                     "Method": method,
#                     "PSNR": round(psnr, 2),
#                     "SSIM": round(ssim, 4)
#                 }
#                 results.append(new_row)
#     df = pd.DataFrame(results)
#     df["Value"] = (
#         df["PSNR"].map("{:.2f}".format)
#         + "/"
#         + df["SSIM"].map("{:.4f}".format)
#     )
#     table = df.pivot(
#         index=["Dataset", "Factor"],
#         columns="Method",
#         values="Value"
#     )
#     table = table.reset_index()

#     table["Set_num"] = table["Dataset"].str.extract(r"Set(\d+)").astype(int)
#     table = table.sort_values(by=["Set_num", "Factor"])
#     table = table.drop(columns="Set_num")
#     table = table[["Dataset", "Factor", "nearest", "bilinear", "bicubic"]]
#     return table


def get_models(models_names, factor, device='cuda'):
    models = []
    for name in models_names:
        if "nearest" == name:
            models.append(Upsampler(scale_factor=factor, mode='nearest'))
            
        elif "bilinear" == name:
            models.append(Upsampler(scale_factor=factor, mode='bilinear'))

        elif "bicubic" == name:
            models.append(Upsampler(scale_factor=factor, mode='bicubic'))

        elif "SRCNN" == name:
            srcnn_weights, _ = load_results("srcnn", factor, ft=False, device=device)
            if srcnn_weights:
                srcnn_model = SRCNN(upsample_factor=factor, middle_k_size=3)
                srcnn_model.load_state_dict(srcnn_weights)
                models.append(srcnn_model)
            else:
                models.append(None)

        elif "FSRCNN" == name:
            fsrcnn_weights, _ = load_results("fsrcnn", factor, ft=False, device=device)
            if fsrcnn_weights:
                fsrcnn_model = FSRCNN(upsample_factor=factor)
                fsrcnn_model.load_state_dict(fsrcnn_weights)
                models.append(fsrcnn_model)
            else:
                models.append(None)

        elif "FSRCNN ft" == name:
            fsrcnn_ft_weights, _ = load_results("fsrcnn", factor, ft=True, device=device)
            if fsrcnn_ft_weights:
                fsrcnn_ft_model = FSRCNN(upsample_factor=factor)
                fsrcnn_ft_model.load_state_dict(fsrcnn_ft_weights)
                models.append(fsrcnn_ft_model)
            else:
                models.append(None)

        elif "FSRCA PS" == name:
            fsrca_weights, _ = load_results("fsrca_ps", factor, ft=False, device=device)
            if fsrca_weights:
                fsrca_model = FSRCA_PS(upsample_factor=factor)
                fsrca_model.load_state_dict(fsrca_weights)
                models.append(fsrca_model)
            else:
                models.append(None)

        elif "FSRCA PS ft" == name:
            fsrca_weights, _ = load_results("fsrca_ps", factor, ft=True, device=device)
            if fsrca_weights:
                fsrca_model = FSRCA_PS(upsample_factor=factor)
                fsrca_model.load_state_dict(fsrca_weights)
                models.append(fsrca_model)
            else:
                models.append(None)

        elif "FSRCA" == name:
            fsrca_weights, _ = load_results("fsrca", factor, ft=False, device=device)
            if fsrca_weights:
                fsrca_model = FSRCA(upsample_factor=factor)
                fsrca_model.load_state_dict(fsrca_weights)
                models.append(fsrca_model)
            else:
                models.append(None)
        
        elif "FSRCA ft" == name:
            fsrca_ft_weights, _ = load_results("fsrca", factor, ft=True, device=device)
            if fsrca_ft_weights:
                fsrca_ft_model = FSRCA(upsample_factor=factor)
                fsrca_ft_model.load_state_dict(fsrca_ft_weights)
                models.append(fsrca_ft_model)
            else:
                models.append(None)
        else:
            raise ValueError(f"Unknown model name: {name}")
    return models
            
def get_models_results(models_name, set_numbers, factors, device="cuda"):
    rows = []
    for factor in factors:
        models = get_models(models_name, factor, device=device)

        for set_number in set_numbers:
            row = {
                "Dataset": f"Set{set_number}",
                "Factor": factor
            }

            for model, title in zip(models, models_name):
                if model:
                    start = time.time()
                    results = evaluate_model_on_sets(
                        model,
                        [set_number],  
                        [factor],     
                        device,
                        print=False,
                        show_plots=False
                    )
                    elapsed = time.time() - start
                    psnr = results[set_number]["psnr"]
                    ssim = results[set_number]["ssim"]

                    row[title] = f"{psnr:.2f}/{ssim:.4f}/{elapsed:.2f}s"
                else:
                    row[title] = "-"
            rows.append(row)

    df = pd.DataFrame(rows)
    df = df[["Dataset", "Factor"] + models_name]

    return df

def overlay_zoom_patch(img, zoom=2, patch_ratio=0.25, margin_ratio=0.05, thickness=2):
    img = np.array(img)

    h_img, w_img = img.shape[:2]
    patch_size = int(min(h_img, w_img) * patch_ratio)
    patch_size = max(1, patch_size // 2 * 2)

    margin_x = int(w_img * margin_ratio)
    margin_y = int(h_img * margin_ratio)
    top  = margin_y
    left = w_img - patch_size - margin_x

    img_patch = img[top:top + patch_size, left:left + patch_size]
    
    zoomed_patch = np.repeat(np.repeat(img_patch, zoom, axis=0),zoom, axis=1)

    overlay = img.copy()
    cv2.rectangle(overlay, (left, top), (left + patch_size, top + patch_size), (255, 0, 0), thickness=thickness)

    h_lr, w_lr = img.shape[:2]
    h_zp, w_zp = zoomed_patch.shape[:2]
    dst_x = w_lr - w_zp
    dst_y = h_lr - h_zp

    overlay[dst_y:dst_y + h_zp, dst_x:dst_x + w_zp] = zoomed_patch

    cv2.rectangle(overlay, (dst_x, dst_y), (dst_x + w_zp, dst_y + h_zp), (255, 0, 0), thickness=thickness)

    return overlay

def plot_zoom_patches(set_dict, img_index, factor, models, models_names, zoom_patches_args, device='cuda'):
    lr, hr = set_dict['lr'][img_index], set_dict['hr'][img_index]
    hr_patch = overlay_zoom_patch(hr, **zoom_patches_args)

    hr_y = get_y_tensors([hr])[0].to(device)  # (1, 1, H, W)
    hr_y_crop = hr_y[:, :, factor:-factor, factor:-factor]  # (1, 1, H', W')
    hr_y_crop = hr_y_crop.squeeze(0).squeeze(0)  # (H', W')

    lr_channels = rgb_to_ycbcr(lr)
    lr_y = lr_channels[0].to(device)  # (1, H, W)
    lr_cbcr = lr_channels[1:]
    
    patches= []
    psnr_list = []

    
    for model in (models):
        model= model.to(device).eval()
        with torch.no_grad():
            sr_y = model(lr_y)
            sr_y = torch.clamp(sr_y.cpu(), 0, 1)
            sr_y = sr_y[:, :, factor:-factor, factor:-factor] # (1, 1, H', W')
            sr_y = sr_y.squeeze(0).squeeze(0)  # (H', W')

        psnr = round(calculate_psnr(hr_y_crop, sr_y, data_range=1.0), 2)
        psnr_list.append(f"{psnr} dB")


        sr_color = color_sr(sr_y, lr_cbcr, factor)
        sr_patch = overlay_zoom_patch(sr_color, **zoom_patches_args)
        patches.append(sr_patch)

    patches.append(hr_patch)
    psnr_list.append('PSNR')

    subtitles =  models_names + ['High Resolution']
    subtitles = [f"{name} / {psnr}" for name, psnr in zip(subtitles, psnr_list)]
    fig, axes = plt.subplots(1, len(patches), figsize=(12, 6))
    for ax, patch, subtitle in zip(axes, patches, subtitles):
        ax.imshow(patch)
        ax.text(
                0.5,                 
                -0.01,               
                subtitle,            
                transform=ax.transAxes,
                ha='center',         
                va='top',            
                fontsize='large'    
            )
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def compare_plot_patches_models(set_dict, img_index, factor, models_names, zoom_patches_args, device):
    models = get_models(models_names, factor, device=device)

    plot_zoom_patches(set_dict, img_index, factor, models, models_names, zoom_patches_args, device=device)

def train_model(model, train_loader, val_loader, factor, num_epochs=50, lr=1e-4, criterion=nn.MSELoss(), early_stopping_patience=10):
    device = torch.device("cuda")
    model.to(device)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)   
    sch_patience = early_stopping_patience // 2
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=sch_patience)
    scaler = GradScaler('cuda')

    train_loss = []
    valid_loss = []
    psnr_list = []

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    checkpoint_path = "checkpoint_best_model.pth"
    checkpoint_freq = 10  # Save checkpoint every 10 epochs

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
            lr_batch, hr_batch = batch
            lr_batch, hr_batch = lr_batch.to(device), hr_batch.to(device)

            optimizer.zero_grad()
            with autocast('cuda'):
                sr_batch = model(lr_batch)
                sr_batch_cropped = sr_batch[:, :, factor:-factor, factor:-factor]
                hr_batch_cropped = hr_batch[:, :, factor:-factor, factor:-factor]
                loss = criterion(sr_batch_cropped, hr_batch_cropped)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.6f}")
        train_loss.append(avg_train_loss)

        # Validación
        model.eval()
        val_loss = 0.0
        psnr_total = 0.0
        val_image_count = 0

        with torch.no_grad():
            for val_batch in val_loader:
                lr_val, hr_val = val_batch
                lr_val, hr_val = lr_val.to(device), hr_val.to(device)
                with autocast('cuda'):
                    sr_val = model(lr_val)
                    sr_val_cropped = sr_val[:, :, factor:-factor, factor:-factor]
                    hr_val_cropped = hr_val[:, :, factor:-factor, factor:-factor]
                    loss = criterion(sr_val_cropped, hr_val_cropped)
                val_loss += loss.item()

                val_image_count += sr_val.shape[0]

                psnr_total += calculate_psnr_torch(hr_val_cropped, sr_val_cropped).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        avg_psnr = psnr_total / val_image_count
        print(f"[Epoch {epoch+1}] Validation Loss: {avg_val_loss:.6f}, PSNR: {avg_psnr:.2f}")
        valid_loss.append(avg_val_loss)
        psnr_list.append(avg_psnr)

        scheduler.step(avg_val_loss)

        # Save checkpoint every 10 epochs if best so far
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
            if (epoch + 1) % checkpoint_freq == 0:
                torch.save(best_model_state, checkpoint_path)
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epochs.")

        if patience_counter >= early_stopping_patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Delete checkpoint if it exists
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    return train_loss, valid_loss, psnr_list

def save_results(model, train_loss, valid_loss, psnr_list, model_name, factor, ft=False):
    model_upper = model_name.upper()
    weights_path = f"../Results/Weights/{model_upper}/{model_name}_x{factor}"
    losses_path = f"../Results/Losses/{model_upper}/{model_name}_x{factor}"
    if ft:
        weights_path += "_ft"
        losses_path += "_ft"

    weights_path += ".pth"
    losses_path += ".json"

    torch.save(model.state_dict(), weights_path)
    with open(losses_path, "w") as f:
        json.dump({"train_loss": train_loss, "valid_loss": valid_loss, "psnr_list": psnr_list}, f)

def compare_loss_plot(loss, models_names, fts, factor, titles, colors, loss_label, device="cuda"):
    """
    Compare the loss of different models.
    """
    losses = []
    for model_name, ft in zip(models_names, fts):
        model_loss = load_results(model_name, factor, ft=ft, device=device)[1][loss]
        losses.append(model_loss)
    
    plt.figure(figsize=(10, 6))
    for model_loss, title, color in zip(losses, titles, colors):
        plt.plot(model_loss, label=f"{title} ({model_loss[-1]:.2f} dB)", color=color)

    plt.xticks(fontsize=14)  
    plt.yticks(fontsize=14)   
    plt.xlabel('Epochs', fontsize=16)
    plt.ylabel(loss_label, fontsize=16)
    plt.legend(fontsize=16)
    plt.grid()
    plt.show()

def load_results(model_name, factor, ft=False, device='cuda'):

    model_upper = model_name.upper()
    weights_path = f"../Results/Weights/{model_upper}/{model_name}_x{factor}"
    losses_path = f"../Results/Losses/{model_upper}/{model_name}_x{factor}"
    if ft:
        weights_path += "_ft"
        losses_path += "_ft"

    weights_path += ".pth"
    losses_path += ".json"
    if not os.path.exists(weights_path):
        print(f"Model weights not found at {weights_path}.")
        return None, None
    model_weights = torch.load(weights_path, map_location=device)
    with open(losses_path, "r") as f:
        losses_data = json.load(f)

    return model_weights, losses_data

def rgb_to_ycbcr(img):
    """ Convert a PIL RGB image to YCbCr and return all channels as tensors. """
    to_tensor = T.ToTensor()
    ycbcr = img.convert("YCbCr").split()  # Extrae sólo el canal Y como PIL.Image
    tensors = [to_tensor(channel).unsqueeze(0) for channel in ycbcr]  # (1, H, W) for each channel
    return tensors

def color_sr(sr, color_channels, f):
    """
    Upsample the color channels by f
    Crop the borders to match the SR image size
    Combine the Y channel with the upsampled color channels
    Convert back to RGB and return as numpy array.
    """
    # Upsample Cb, Cr
    upsampled = [F.interpolate(channel, scale_factor=f, mode='bicubic', align_corners=False) for channel in color_channels]
    # Crop borders
    cropped = [channel[:, :, f:-f, f:-f] for channel in upsampled]
    # Convert tensors to PIL images
    to_pil = T.ToPILImage()
    y_pil = to_pil(sr.unsqueeze(0))  # (H, W) -> PIL
    cb_pil = to_pil(cropped[0].squeeze(0))
    cr_pil = to_pil(cropped[1].squeeze(0))
    # Recombinar como YCbCr
    ycbcr = Image.merge("YCbCr", (y_pil, cb_pil, cr_pil))
    # Convertir a RGB
    rgb = ycbcr.convert("RGB")
    return np.array(rgb)


def evaluate_model_on_sets(model, sets, factor_list, device, print=True, show_plots=True):
    test_imgs = get_test_images(sets, factor_list)
    model.eval()
    results = {s: {} for s in sets}
    for s in sets:
        for f in factor_list:
            lr_imgs = test_imgs[s][f]["lr"]
            hr_imgs = test_imgs[s][f]["hr"]

            # Convert HR PIL images to Y channel tensors and crop
            hr_y_imgs = [img[:, :, f:-f, f:-f].squeeze(0).squeeze(0) for img in get_y_tensors(hr_imgs)]
            sr_y_imgs = []
            sr_color_imgs_np = []

            for img in lr_imgs:
                img_channels = rgb_to_ycbcr(img)
                y_channel = img_channels[0].to(device)  # (1, H, W)
                with torch.no_grad():
                    sr_y = model(y_channel)
                sr_y = torch.clamp(sr_y.cpu(), 0, 1)
                sr_y = sr_y[:, :, f:-f, f:-f].squeeze(0).squeeze(0)
                sr_y_imgs.append(sr_y)
                sr_color_imgs_np.append(color_sr(sr_y, img_channels[1:], f))

            # Convert to numpy for metrics and plotting
            sr_y_imgs_np = [img.numpy() for img in sr_y_imgs]
            hr_y_imgs_np = [img.numpy() for img in hr_y_imgs]
            hr_color_imgs_np = [np.array(img) for img in hr_imgs]
            # Plot matrix if requested
            if show_plots:
                n = len(lr_imgs)
                fig, axs = plt.subplots(n, 3, figsize=(12, 4 * n))
                if n == 1:
                    axs = [axs]
                for i in range(n):
                    axs[i][0].imshow(np.array(lr_imgs[i]), cmap='gray')
                    axs[i][0].set_title('Low Resolution')
                    axs[i][0].axis('off')
                    axs[i][1].imshow(sr_color_imgs_np[i], cmap='gray')
                    axs[i][1].set_title('Super Resolution')
                    axs[i][1].axis('off')
                    axs[i][2].imshow(hr_color_imgs_np[i], cmap='gray')
                    axs[i][2].set_title('High Resolution')
                    axs[i][2].axis('off')
                plt.tight_layout()
                plt.show()

            # Calculate metrics for this set
            psnr, ssim = calculate_average_metrics(hr_y_imgs_np, sr_y_imgs_np, data_range=1.0)
            psnr = round(psnr, 2)
            ssim = round(ssim, 4)
            
            results[s]= {
                "psnr": psnr,
                "ssim": ssim
            }
    if print:
        for s, metrics in results.items():
            print(f"Set {s} - Average PSNR: {metrics['psnr']} dB, SSIM: {metrics['ssim']}")
    else:
        return results
