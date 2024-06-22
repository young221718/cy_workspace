import torch
from torchvision import datasets, models, transforms

import torch.nn as nn
import torch.optim as optim

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transforms
transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


print("Start")
# Load ImageNet dataset
train_dataset = datasets.ImageNet(
    root="/home/work/Dataset/imagenet", split="train", transform=transform
)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=32, shuffle=True, num_workers=4
)
print("train_dataset:", len(train_dataset))
# Load validation dataset
val_dataset = datasets.ImageNet(
    root="/home/work/Dataset/imagenet", split="val", transform=transform
)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=32, shuffle=False, num_workers=4
)
print("val_dataset:", len(val_dataset))

# Define the model
model = models.swin_t()
# Move the model to the device
model = model.to(device)
# Define the loss function
criterion = nn.CrossEntropyLoss()
# Define the optimizer
optimizer = optim.AdamW(model.parameters(), lr=0.0001)

min_val_loss = 100000
# Train the model
for epoch in range(100):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 100 == 99:
            print(f"[Epoch {epoch+1}, Batch {i+1}] loss: {running_loss/100:.3f}")
            running_loss = 0.0
    # Validate the model
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data in val_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Validation loss: {val_loss/len(val_loader):.3f}")
    print(f"Validation accuracy: {(100 * correct / total):.2f}%")

    # Save the model if validation loss is minimized
    if val_loss < min_val_loss:
        min_val_loss = val_loss
        torch.save(
            model.state_dict(),
            "/home/work/StudentsWork/ChanYoung/cy_workspace/classification/best_model.pth",
        )
        print("Saved best model")

print("Training and validation finished")
