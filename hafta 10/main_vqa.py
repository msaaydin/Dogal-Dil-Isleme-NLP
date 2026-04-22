import numpy as np 
import torch
import torchvision
import torchvision.transforms as T
from torchvision.transforms import functional as F
from PIL import Image
from easy_vqa import get_train_questions, get_test_questions, get_train_image_paths, get_test_image_paths, get_answers
import pandas as pd
import torchvision.utils as utils
from torchvision import transforms
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

# sentence_transformers ve  easy_vqa kütüphanelerininin kurulumu için: pip install easy-vqa, pip install -U sentence-transformers




def load_and_process_image(image_path):
    # Loads image from path and converts to Tensor, you can also reshape the im
    im = Image.open(image_path)
    im = F.to_tensor(im)
    return im

def read_images(paths):
    # paths is a dict mapping image ID to image path
    # Returns a dict mapping image ID to the processed image
    ims = {}
    for image_id, image_path in paths.items():
        ims[image_id] = load_and_process_image(image_path)
    return ims

print('--- Reading/processing images from image paths of the vqa library ---\n')
train_ims = read_images(get_train_image_paths())
test_ims = read_images(get_test_image_paths())
im_shape = train_ims[0].shape
print(f'Read {len(train_ims)} training images and {len(test_ims)} testing images.')
print(f'Each image has shape {im_shape}.')

print('\n--- Creating model input images...')
train_qs, train_answers, train_image_ids = get_train_questions()
test_qs, test_answers, test_image_ids = get_test_questions()
train_X_ims = np.array([train_ims[id] for id in train_image_ids])
test_X_ims = np.array([test_ims[id] for id in test_image_ids])

print('\n--- Reading questions...')
train_qs, train_answers, train_image_ids = get_train_questions()
test_qs, test_answers, test_image_ids = get_test_questions()
print(f'Read {len(train_qs)} training questions and {len(test_qs)} testing questions.')

print('\n--- Reading answers...')
all_answers = get_answers()
num_answers = len(all_answers)
print(f'Found {num_answers} total answers:')
print(all_answers)

df = pd.DataFrame(list(zip(train_qs, train_answers, train_image_ids)), columns =['Question', 'Answer', 'Image ID'])
print(df.head(10))





# print multiple images
# images = 1
# batch = torch.empty((images, 3, 64, 64))
# for i in range(images):
#     batch[i] = train_ims[i]

# Create a grid of images
id = 0
grid = utils.make_grid(train_ims[id], nrow=2)
# Convert the grid to a PIL image
image = transforms.ToPILImage()(grid)
# Show the image
image.show()


st_model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')

#Questions are encoded by calling model.encode()
train_X_seqs = st_model.encode(train_qs)
test_X_seqs = st_model.encode(test_qs)

# convert ndarray to tensor    
train_X_seqs = torch.tensor(train_X_seqs, dtype=torch.float) 
test_X_seqs = torch.tensor(test_X_seqs, dtype=torch.float)

print(f'\nThe shape of the binary vectors is : {train_X_seqs.shape}')
print('\n--- Creating model outputs...')

train_answer_indices = np.array([all_answers.index(a) for a in train_answers])
test_answer_indices = np.array([all_answers.index(a) for a in test_answers])

#creating a 2D array filled with 0's
train_Y = np.zeros((train_answer_indices.size, train_answer_indices.max()+1), dtype=int)
test_Y = np.zeros((test_answer_indices.size, test_answer_indices.max()+1), dtype=int)

#replacing 0 with a 1 at the index of the original array
train_Y[np.arange(train_answer_indices.size),train_answer_indices] = 1
test_Y[np.arange(test_answer_indices.size),test_answer_indices] = 1 

# finally convert the label vectors to tensor and fix the data type so it wouldnt error in the fully connected layer
train_Y = torch.tensor(train_Y, dtype=torch.float)
test_Y = torch.tensor(test_Y, dtype=torch.float)

print(f'Example model output: {train_Y[0]}')
print(f'data type {type(train_Y)}')

import torch
import torchvision
from torch import mul, cat, tanh, relu

class VQA_v2(torch.nn.Module):
  def __init__(self, embedding_size, num_answers):
    super(VQA_v2, self).__init__()

    # The Image network which processes image and outputs a vector of shape (batch_size x 32) 
    resnet = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
    num_ftrs = resnet.fc.in_features
    resnet.fc = torch.nn.Sequential(
        torch.nn.Linear(num_ftrs, 512),
        torch.nn.Tanh(),
        torch.nn.Linear(512, 128),
        torch.nn.Tanh(),
        torch.nn.Linear(128, 32)
    )

    self.mdl = resnet

    # The question network processes the question and outputs a vector of shape (batch_size x 32)
    self.fc2 = torch.nn.Linear(embedding_size, 64)      # (384, 64)
    self.fc3 = torch.nn.Linear(64, 32)                  # (64, 32)

    # Layers for Merging operation
    self.fc4 = torch.nn.Linear(64, 32)                  
    self.fc5 = torch.nn.Linear(32, num_answers)

  def forward(self, x, q):
    # The Image network
    x = self.mdl(x)                             # (batch_size, 32)

    # The question network
    act = torch.nn.Tanh() 
    q = act(self.fc2(q))                        # (32, 32)
    q = act(self.fc3(q))                        # (32, 32)

    # Merge -> output                              
    out = cat((x, q), 1)                        # concat function
    out = act(self.fc4(out))                    # activation
    out = self.fc5(out)                         # output probability
    return out
  


class CustomDataset(Dataset):
    def __init__(self, img, txt, ans):
      self.img = img
      self.txt = txt
      self.ans = ans

    def __len__(self):
        return len(self.ans)

    def __getitem__(self, idx):
      ans = self.ans[idx]
      img = self.img[idx]
      txt = self.txt[idx]
      return img, txt, ans
    

def train_loop(model, optimizer, criterion, train_loader):
    model.train()
    model.to(device)
    total_loss, total = 0, 0

    for image, text, label in trainloader:
        # get the inputs; data is a list of [inputs, labels]
        image, text, label =  image.to(device), text.to(device), label.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        output = model.forward(image, text)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        # Record metrics
        total_loss += loss.item()
        total += len(label)

    return total_loss / total


def validate_loop(model, criterion, valid_loader):
    model.eval()
    model.to(device)
    total_loss, total = 0, 0

    with torch.no_grad():
      for image, text, label in testloader:
          # get the inputs; data is a list of [inputs, labels]
          image, text, label =  image.to(device), text.to(device), label.to(device)

          # Forward pass
          output = model.forward(image, text)

          # Calculate how wrong the model is
          loss = criterion(output, label)

          # Record metrics
          total_loss += loss.item()
          total += len(label)

    return total_loss / total



from torch.utils.data import DataLoader

from tqdm.notebook import tqdm



# WandB – Config is a variable that holds and saves hyperparameters and inputs
batch_size = 32         # input batch size for training (default: 64)
test_batch_size = 32    # input batch size for testing (default: 1000)
epochs = 40            # number of epochs to train (default: 10)
lr = 0.01               # learning rate (default: 0.01)
momentum = 0.5          # SGD momentum (default: 0.5) 
no_cuda = False         # disables CUDA training
log_interval = 10     # how many batches to wait before logging training status


if torch.cuda.is_available(): device = torch.device("cuda:0")
kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}


# Now we load our training and test datasets initialize the train, validation, and test data loaders

train_dataset = CustomDataset(train_X_ims, train_X_seqs, train_Y)
test_dataset = CustomDataset(test_X_ims, test_X_seqs, test_Y)
trainloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
testloader = DataLoader(test_dataset, batch_size=test_batch_size)


# Initialize our model, recursively go over all modules and convert their parameters and buffers to CUDA tensors (if device is set to cuda)
model = VQA_v2(embedding_size = 384, num_answers = num_answers).to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr )

# WandB – wandb.watch() automatically fetches all layer dimensions, gradients, model parameters and logs them automatically to your dashboard.
# Using log="all" log histograms of parameter values in addition to gradients

train_losses, valid_losses = [], []

for epoch in range(epochs):
    train_loss = train_loop(model, optimizer, criterion, trainloader)
    valid_loss = validate_loop(model, criterion, testloader)
    
    tqdm.write(
        f'epoch #{epoch + 1:3d}\ttrain_loss: {train_loss:.2e}\tvalid_loss: {valid_loss:.2e}\n',
    )
    
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)

    print({"Epoch": epoch, "Training Loss": train_loss, "Validation Loss": valid_loss})
    
plt.style.use('ggplot')


epoch_ticks = range(1, epoch + 2)
plt.plot(epoch_ticks, train_losses)
plt.plot(epoch_ticks, valid_losses)
plt.legend(['Train Loss', 'Valid Loss'])
plt.title('Losses') 
plt.xlabel('Epoch #')
plt.ylabel('Loss')
plt.xticks(epoch_ticks)
plt.show()


model.eval()
model.to(device)
num_correct = 0
num_samples = 0
predictions = []
answers = []

with torch.no_grad():
    for image, text, label in testloader:
        image, text, label =  image.to(device), text.to(device), label.to(device)
        probs = model.forward(image, text)

        _, prediction = probs.max(1)
        predictions.append(prediction)      

        answer = torch.argmax(label, dim=1) 
        answers.append(answer)

        num_correct += (prediction == answer).sum()
        num_samples += prediction.size(0)
        
    valid_acc = (f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}')     
    print(valid_acc)

    print({"Validation Accuracy": round(float(num_correct)/float(num_samples)*100, 2)})  

torch.save(model.state_dict(), 'Resnet-Sbert-40.pth')

model = VQA_v2(embedding_size = 384, num_answers = 13)
model.load_state_dict(torch.load('Resnet-Sbert-40.pth')) # Im loading my best model
model.eval()

from urllib.request import urlopen
from PIL import Image

def load_and_process_image_url(url):
    # Loads image from path and converts to Tensor, you can also reshape the im
    im = Image.open(urlopen(url))
    im = F.to_tensor(im)
    return im


url = "https://www.nicepng.com/png/detail/16-163438_circle-clipart-sky-blue-clip-art-blue-circle.png"
image = load_and_process_image_url(url)
image = image.unsqueeze(0)

text = 'What shape is this?'
text = st_model.encode(text)
text = torch.tensor(text, dtype=torch.float)
text = text.unsqueeze(0)

probs = model.forward(image, text)
answer_idx = torch.argmax(probs, dim=1) # get index of answer with highest probability
answer_text = [all_answers[idx] for idx in answer_idx] # convert index to answer text
print(answer_text)