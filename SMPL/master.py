from data_handling import get_cifar10_data, load_batches, create_classification_directory, Copy_train_Images, \
    Images_in_directory, Copy_test_Images, simple_dataset, get_subdirectories, l_u_split
from data_transformation import get_transform
from data_loading import get_loaders
from model import Wide_ResNet
from train import train_model, evaluate_model

# Hyperparameters and device
BATCH_SIZE = 128
labeled_dir = 'labeled'
unlabeled_dir = 'unlabeled'
test_dir = 'test'
device = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
EPOCHS = 100
BETA_ZERO = 8

# Step 1: Data Handling
train_data, test_data = get_cifar10_data()
meta_path = 'data/cifar-10-batches-py/batches.meta'
meta = load_batches(meta_path)

# Step 2: Data Transformation
transform = get_transform()

# Step 3: Data Loading
labeled_train_loader, unlabeled_train_loader_augmented, unlabeled_train_loader, test_loader = get_loaders(
    BATCH_SIZE, labeled_dir, unlabeled_dir, test_dir, transform, simple_transform
)

# Step 4: Model Definition
model = Wide_ResNet().to(device)

# Step 5: Training
optimizer = torch.optim.SGD(model.parameters(), lr=0.05, weight_decay=1e-4, momentum=0.9)
criterion = nn.CrossEntropyLoss()

train_model(model, labeled_train_loader, unlabeled_train_loader_augmented, optimizer, criterion, device)

# Step 6: Evaluation
evaluate_model(model, test_loader, criterion, device)
