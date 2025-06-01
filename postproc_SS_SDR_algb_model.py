import numpy as np
from matplotlib import pyplot as plt
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Load and preprocess data
with open('./FisierData/data_SS_SDRmeanSDRflSDR_components.pickle', 'rb') as f:
    data1 = pickle.load(f, encoding='latin1')

x = data1['x']
r = data1['r']
SDR = data1['SDR']
SDRmean = data1['SDRmean']
SDRfl = data1['SDRfl']

with open('./FisierData/data_all_means.pickle', 'rb') as f:
    data2 = pickle.load(f, encoding='latin1')

u0 = data2['u0']
Z0 = data2['Z0']
v0 = data2['v0']

with open('./FisierData/data_SS_EpstkeZ0flZ0fl.pickle', 'rb') as f:
    data3 = pickle.load(f, encoding='latin1')

eps = data3['eps']
tke = data3['tke']
Z0flZ0fl = data3['Z0flZ0fl']

u = u0[:2581,:570]
Z = Z0[:2581,:570]

x = x*2
r  = r*2

itr_ran = np.arange(762500, 902001, 500)
itr = 762500
x_D = np.zeros((x.size))

U0 = 1
x0 = 2.39
D = 2
tau = D / U0
dt = 0.001
t = dt * (itr_ran-itr)
t_tau = t/tau

for i in range(0, len(x)):
    x_D[i] = (x[i] - x0*D) / D

end = 2579

ua = u[:,0]
Z0a = Z0[:,0]
x = x[:2581]
r = r[:570]
r_1_2 = np.zeros((u[:,0].size),dtype='int')
for i in range(0,u[:,0].size):
        r_1_2[i] = np.int64(np.argmin(np.abs(0.5*u[i,0]-u[i,:])))

r_1_2a = r[r_1_2[:]]

x_D = np.zeros((x.size))
for i in range(0,x.size):
    x_D[i] = (x[i]-x0*D)/D

diff = 3.81 * 10**(-4)

epsSS = np.zeros((2581,570))
tkeSS = np.zeros((2581,570))
Z0flZ0flSS = np.zeros((2581,570))
SDRalgb = np.zeros((2581,570))

C = 2.0
nu = 0.72*diff

for i in range(0,eps[:,0].size):
        epsSS[i,:] = 2*nu*eps[i,:]
        tkeSS[i,:] = 0.5*tke[i,:]
        Z0flZ0flSS[i,:] = Z0flZ0fl[i,:]
        SDRalgb[i,:] = C * epsSS[i,:]/tkeSS[i,:] * Z0flZ0flSS[i,:]

idx = 2

eta_avg = np.zeros((570))

start = 2400
for i in range(start,2579):
    eta_avg = eta_avg + r[:]/(x[i]-x0*D)

eta_avg = eta_avg/(2579-start)

SDRn = np.zeros((SDR[:,0].size, SDR[0,:].size))
SDRalgbn = np.zeros((SDR[:,0].size, SDR[0,:].size))

for i in range(0,2580):
    SDRn[i,:] = SDR[i,:]/SDR[i,idx]
    SDRalgbn[i,:] = SDRalgb[i,:]/SDRalgb[i,15]

SDRstd = np.std(SDRn[start:2579,:],0)
SDRm = np.mean(SDRn[start:2579,:],0)
SDRalgbm = np.mean(SDRalgbn[start:2579,:],0)

# Preprocessing for PyTorch
epsSS[epsSS == 0] = 1e-10
tkeSS[tkeSS == 0] = 1e-10
Z0flZ0flSS[Z0flZ0flSS == 0] = 1e-10

# Normalization
epsSS_mean, epsSS_std = np.mean(epsSS[start:2579, :]), np.std(epsSS[start:2579, :]) + 1e-10
tkeSS_mean, tkeSS_std = np.mean(tkeSS[start:2579, :]), np.std(tkeSS[start:2579, :]) + 1e-10
Z0fl_mean, Z0fl_std = np.mean(Z0flZ0flSS[start:2579, :]), np.std(Z0flZ0flSS[start:2579, :]) + 1e-10
SDRn_mean, SDRn_std = np.mean(SDRn[start:2579, :]), np.std(SDRn[start:2579, :]) + 1e-10

epsSS_norm = (epsSS[start:2579, :] - epsSS_mean) / epsSS_std
tkeSS_norm = (tkeSS[start:2579, :] - tkeSS_mean) / tkeSS_std
Z0fl_norm = (Z0flZ0flSS[start:2579, :] - Z0fl_mean) / Z0fl_std
SDRn_norm = (SDRn[start:2579, :] - SDRn_mean) / SDRn_std

# Derived feature: (epsSS/tkeSS) * Z0flZ0flSS
derived_feature = (epsSS / tkeSS) * Z0flZ0flSS
derived_mean, derived_std = np.mean(derived_feature[start:2579, :]), np.std(derived_feature[start:2579, :]) + 1e-10
derived_norm = (derived_feature[start:2579, :] - derived_mean) / derived_std

# Prepare data
X_eps = epsSS_norm.reshape(-1, 1)
X_tke = tkeSS_norm.reshape(-1, 1)
X_Z0fl = Z0fl_norm.reshape(-1, 1)
X_derived = derived_norm.reshape(-1, 1)
y = SDRn_norm.reshape(-1, 1)

X = np.hstack((X_eps, X_tke, X_Z0fl, X_derived))

# Manual train-validation split (80-20 split)
n_samples = X.shape[0]
n_train = int(0.8 * n_samples)
indices = np.arange(n_samples)
np.random.shuffle(indices)

train_indices = indices[:n_train]
val_indices = indices[n_train:]

X_train, X_val = X[train_indices], X[val_indices]
y_train, y_val = y[train_indices], y[val_indices]

# Convert to PyTorch tensors
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)
X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

# DataLoader for batch training
dataset = TensorDataset(X_train_tensor, y_train_tensor)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

# PyTorch model
class SDRPredictor(nn.Module):
    def __init__(self):
        super(SDRPredictor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(4, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)

# Initialize model, loss, and optimizer
model = SDRPredictor().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=20)

# Train with early stopping
num_epochs = 1000
best_val_loss = float('inf')
patience = 150
counter = 0

for epoch in range(num_epochs):
    model.train()
    for batch_X, batch_y in dataloader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_tensor)
        val_loss = criterion(val_outputs, y_val_tensor)

    scheduler.step(val_loss)

    if (epoch + 1) % 50 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')

    if val_loss.item() < best_val_loss:
        best_val_loss = val_loss.item()
        torch.save(model.state_dict(), 'best_model.pth')
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping triggered")
            break

# Load best model
model.load_state_dict(torch.load('best_model.pth'))

# Predictions
model.eval()
with torch.no_grad():
    predictions_norm = model(X_tensor).cpu().numpy()

# Denormalize predictions
predictions = predictions_norm * SDRn_std + SDRn_mean
predictions = predictions.reshape(2579 - start, 570)
pred_mean = np.mean(predictions, axis=0)

# Plot
plt.figure(figsize=(14, 10))
plt.fill_between(eta_avg[:], (SDRm - SDRstd), (SDRm + SDRstd), alpha=0.3, color='b',
                 label='Normalised SDR (Ground Truth)')
plt.plot(eta_avg[:], SDRm, linestyle='-', color='b', linewidth=4)
plt.plot(eta_avg[:], pred_mean, linestyle='--', color='r', linewidth=4, label='PyTorch Prediction')
plt.xlim([0, 0.25])
plt.ylim([0, 1.2])
plt.ylabel('Normalised SDR')
plt.legend()
plt.savefig('sdr_prediction.png')
plt.show()