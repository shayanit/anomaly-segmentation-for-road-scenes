import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import cityscapes
from bisenetv1 import BiSeNetV1  # Assuming you have defined your BiSeNetV1 model class
import argparse
from torchvision import transforms
from main import MyCoTransform

# Define the co-transform for training set
co_transform = MyCoTransform(False, augment=True, height=1024)#1024)
co_transform_val = MyCoTransform(False, augment=False, height=1024)#1024)

# Define command-line arguments
parser = argparse.ArgumentParser(description='BiSeNet Training')
parser.add_argument('--datadir', type=str, default='path/to/data', help='Path to the dataset directory')
parser.add_argument('--num_workers', type=int, default=2, help='Number of workers for data loading')
parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
# Add more arguments as needed

args = parser.parse_args()

if __name__ == '__main__':   

    # Define your dataset and DataLoader
    dataset_train = cityscapes(args.datadir, co_transform, 'train')
    dataset_val = cityscapes(args.datadir, co_transform_val, 'val')

    loader = DataLoader(dataset_train, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True)
    loader_val = DataLoader(dataset_val, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)

    # Define the BiSeNetV1 model
    bisenet = BiSeNetV1(20)  # You need to define your BiSeNetV1 model
    print("Bisenet model initialized!")

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(bisenet.parameters(), lr=0.001)

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bisenet.to(device)

    # Training loop
    num_epochs = 10
    print("Going for the first epoch!")
    for epoch in range(num_epochs):
        # print(f"Epoch: {num_epochs}")
        bisenet.train()
        running_loss = 0.0
        
        for step, (images, labels) in enumerate(loader):
            images, labels = images.to(device), labels.to(device)
            # print(f"label size before squeeze: {labels.size()}")
            labels = labels.squeeze(1)
            # print(f"label size after squeeze: {labels.size()}")
            
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = bisenet(images)
            main_outputs = outputs[0]
            # print(f"main outputs size: {main_outputs.size()}")
            
            # Compute the loss
            loss = criterion(main_outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
        
        epoch_loss = running_loss / len(dataset_train)
        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {epoch_loss:.4f}")
        
        # Validation
        bisenet.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images_val, labels_val in loader_val:
                images_val, labels_val = images_val.to(device), labels_val.to(device)
                labels_val = labels_val.squeeze(1)
                outputs_val = bisenet(images_val)
                main_outputs_val = outputs_val[0]
                val_loss += criterion(main_outputs_val, labels_val).item() * images_val.size(0)
        
        val_loss /= len(dataset_val)
        print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}")

    # Save the trained model
    torch.save(bisenet.state_dict(), 'bisenetv1_model.pth')
    print("Saved!")
