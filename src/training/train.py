def train(model, dataloader, scheduler, optimizer, device, epoch):

    # Single epoch training

    model.train()
    epoch_loss = 0

    progress_bar = tqdm(dataloader, desc="Training")
    for epochs, (high_res_img, low_res_img) in enumerate(progress_bar):
        high_res_img = high_res_img.to(device)
        low_res_img = low_res_img.to(device)

        optimizer.zero_grad()

        # Sample random noise and random timestep

        noise = torch.randn_like(low_res_img).to(device)
        t = torch.randint(0, CONFIG.TIMESTEPS, (low_res_img.shape[0],), device=device)

        noisy_images = scheduler.add_noise(high_res_img,t,noise)

        pred_noise = model(noisy_images, t,low_res_img)

        loss = F.mse_loss(pred_noise,noise)
        epoch_loss += loss.item()

        loss.backward()
        optimizer.step()

        progress_bar.set_postfix(loss=loss.item())

    avg_loss = epoch_loss / len(dataloader)
    print(f"Epoch: {epoch+1} - Average loss: {avg_loss:.4f}")
    return avg_loss


