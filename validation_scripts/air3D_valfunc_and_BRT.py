# Enable import from parent package
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

import dataio, utils, training, loss_functions, modules, diff_operators

import torch
import numpy as np
import math
from torch.utils.data import DataLoader
import configargparse
import scipy.io as spio

logging_root = './logs'
angle_alpha = 1.2

# Setting to plot
ckpt_path = './Deepreach_trained_checkpoints/air3D_ckpt.pth'
activation = 'sine'
times = [0.9]
time_indices_matlab = [int(time_to_plot/0.1) + 1 for time_to_plot in times]
thetas = [1.5863] # This theta is contained in the LS computation grid. 

# Initialize and load the model
model = modules.SingleBVPNet(in_features=4, out_features=1, type=activation, mode='mlp',
                             final_layer_factor=1., hidden_features=512, num_hidden_layers=3)
model.cuda()
checkpoint = torch.load(ckpt_path)
try:
  model_weights = checkpoint['model']
except:
  model_weights = checkpoint
model.load_state_dict(model_weights)
model.eval()

# Load the ground truth BRS data
true_BRT_path = './Deepreach_trained_checkpoints/analytical_BRT_air3D.mat'
true_data = spio.loadmat(true_BRT_path)

# Save the value function arrays
val_functions = {}
val_functions['LS'] = []
val_functions['siren'] = []

# Define the validation function
def val_fn_BRS(model):
  num_times = len(times)  
  num_thetas = len(thetas)

  # Find matlab indices for theta slices
  theta_indices_matlab = []
  theta_values = true_data['gmat'][0, 0, :, 2]
  for i in range(num_thetas):
    theta_indices_matlab.append(np.argmin(abs(theta_values - thetas[i])))

  # Create figures
  fig_brs = plt.figure(figsize=(5*num_thetas, 5*num_times))
  fig_valfunc_LS = plt.figure(figsize=(5*num_thetas, 5*num_times))
  fig_valfunc_siren = plt.figure(figsize=(5*num_thetas, 5*num_times))

  # Start plotting the results
  for i in range(num_times):
    for j in range(num_thetas):
      state_coords = torch.tensor(np.reshape(true_data['gmat'][:, :, theta_indices_matlab[j], :], (-1, 3)), dtype=torch.float32)
      state_coords[:, 2] = state_coords[:, 2] / (angle_alpha * math.pi)
      time_coords = torch.ones(state_coords.shape[0], 1) * times[i]
      coords = torch.cat((time_coords, state_coords), dim=1)[None] 
      
      # Compute the value function
      model_in = {'coords': coords.cuda()}
      model_out = model(model_in)

      # Detatch outputs and reshape
      valfunc = model_out['model_out'].detach().cpu().numpy()
      valfunc_true = true_data['data'][:, :, theta_indices_matlab[j], time_indices_matlab[i]]
      valfunc = np.reshape(valfunc, valfunc_true.shape)

      # Unnormalize the value function and gradients
      norm_to = 0.02
      mean = 0.25
      var = 0.5
      valfunc = (valfunc*var/norm_to) + mean 

      ## Plot the zero level set
      # Fetch the BRS
      brs_predicted = (valfunc <= 0.001) * 1.
      brs_actual = (valfunc_true <= 0.001) * 1.
      # Plot it
      ax = fig_brs.add_subplot(num_times, num_thetas, (j+1) + i*num_thetas)
      ax.set_title('t = %0.2f, theta = %0.2f' % (times[i], thetas[j]))
      s1 = ax.imshow(brs_predicted.T, cmap='bwr', origin='lower', vmin=-1., vmax=1., extent=(-1., 1., -1., 1.), interpolation='bilinear')
      s2 = ax.imshow(brs_actual.T, cmap='seismic', alpha=0.5, origin='lower', vmin=-1., vmax=1., extent=(-1., 1., -1., 1.), interpolation='bilinear')

      ## Plot the actual value function
      ax = fig_valfunc_LS.add_subplot(num_times, num_thetas, (j+1) + i*num_thetas)
      ax.set_title('t = %0.2f, theta = %0.2f' % (times[i], thetas[j]))
      s = ax.imshow(valfunc_true.T, cmap='bwr', origin='lower', extent=(-1., 1., -1., 1.), vmin=-0.25, vmax=1.2)
      fig_valfunc_LS.colorbar(s) 

      ## Plot the predicted value function
      ax = fig_valfunc_siren.add_subplot(num_times, num_thetas, (j+1) + i*num_thetas)
      ax.set_title('t = %0.2f, theta = %0.2f' % (times[i], thetas[j]))
      s = ax.imshow(valfunc.T, cmap='bwr', origin='lower', extent=(-1., 1., -1., 1.), vmin=-0.25, vmax=1.2)
      fig_valfunc_siren.colorbar(s) 

      ## Append the value functions
      val_functions['LS'].append(valfunc_true)
      val_functions['siren'].append(valfunc)

  return fig_brs, fig_valfunc_LS, fig_valfunc_siren, val_functions

# Run the validation of sets
fig_brs, fig_valfunc_LS, fig_valfunc_siren, val_functions = val_fn_BRS(model)

fig_brs.savefig(os.path.join(logging_root, 'Air3D_BRS_comparison.png'))
fig_valfunc_LS.savefig(os.path.join(logging_root, 'Air3D_LS_valfunc.png'))
fig_valfunc_siren.savefig(os.path.join(logging_root, 'Air3D_Siren_valfunc.png'))
spio.savemat(os.path.join(logging_root, 'Air3D_raw_valfuncs.mat'), val_functions)
