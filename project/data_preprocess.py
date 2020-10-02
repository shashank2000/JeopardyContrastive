from torchvision import transforms
from dataset.jeopardy_dataset import JeopardyDataset
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader

# Enter path for questions_file, answers_file and directory for COCO images that correspond to these
questions_file = "../../datasets/questions.json"
answers_file = "../../datasets/answers.json"

# copy and then unarchive instead?
coco_loc = "../../datasets/train2014"

img_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
dataset = JeopardyDataset(questions_file, answers_file, coco_loc, img_transform)

# is it possible that we only have 3 elements in a certain input state?
batch_size = 16
validation_split = .2
shuffle_dataset = True
random_seed= 42

# Creating data indices for training and validation splits:
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(validation_split * dataset_size)

train_indices, val_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = DataLoader(dataset, batch_size=batch_size, 
                                           sampler=train_sampler, num_workers=4, pin_memory=True, drop_last=True)
val_loader = DataLoader(dataset, batch_size=batch_size,
                                                sampler=valid_sampler, num_workers=4, pin_memory=True, drop_last=True)

