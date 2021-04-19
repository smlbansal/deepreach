# Enable import from parent package
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

import dataio, modules, diff_operators

import torch
import numpy as np
import scipy
import math
from torch.utils.data import DataLoader
import configargparse
import scipy.io as spio
import scipy.ndimage

def angle_normalize(x):
  return (((x + np.pi) % (2 * np.pi)) - np.pi)

# Basic parameters
logging_root = './logs'
angle_alpha = 1.2

# Value function level to plot
level = 0.001 

## Setting to plot
ckpt_path = './Deepreach_trained_checkpoints/multivehicle_collision_2E1P.pth'
ckpt_path_1P = './Deepreach_trained_checkpoints/multivehicle_collision_1E1P.pth'

poss = {}
thetas = {}
# Position and theta slices to be plotted for the 1st Evader
poss['1E'] = [(-0.4, 0.0)]
thetas['1E'] = [0.0*math.pi]
# Position and theta slices to be plotted for the 2nd Evader
poss['2E'] = [(0.43, 0.33)]
thetas['2E'] = [-2.44]
# Theta of the ego vehicle
ego_vehicle_theta = [-2.54]

# Time at which the sets should be plotted
time = 1.0

# Number of slices to plot
num_slices = len(poss['1E'])

# Load the model
model = modules.SingleBVPNet(in_features=10, out_features=1, type='sine', mode='mlp', final_layer_factor=1., hidden_features=512, num_hidden_layers=3)
model.cuda()
checkpoint = torch.load(ckpt_path)
try:
  model_weights = checkpoint['model']
except:
  model_weights = checkpoint
model.load_state_dict(model_weights)
model.eval()

# Load the 2 vehicle model
model_1P = modules.SingleBVPNet(in_features=7, out_features=1, type='sine', mode='mlp',
                                final_layer_factor=1., hidden_features=512, num_hidden_layers=3)
model_1P.cuda()
checkpoint = torch.load(ckpt_path_1P)
try:
  model_weights = checkpoint['model']
except:
  model_weights = checkpoint
model_1P.load_state_dict(model_weights)
model_1P.eval()

# Save the value function arrays
val_functions = {}
val_functions['pairwise'] = []
val_functions['full'] = []

def val_fn_BRS_posspace(model, model_1P):
  # Create a figure
  fig = plt.figure(figsize=(5*num_slices, 5))
  fig_error = plt.figure(figsize=(5*num_slices, 5))
  fig_valfunc = plt.figure(figsize=(5*num_slices, 5))

  # Get the meshgrid in the (x, y) coordinate
  sidelen = 200
  mgrid_coords = dataio.get_mgrid(sidelen)

  # Time coordinates
  time_coords = torch.ones(mgrid_coords.shape[0], 1) * time

  # Start plotting the results
  for i in range(num_slices):
    coords = torch.cat((time_coords, mgrid_coords), dim=1) 
    pairwise_coords = {}

    # Setup the X-Y coordinates
    for j in range(2):
      evader_key = '%i' %(j+1) + 'E'
      # X-Y coordinates of the evaders for the full game
      xcoords = torch.ones(mgrid_coords.shape[0], 1) * poss[evader_key][i][0]
      ycoords = torch.ones(mgrid_coords.shape[0], 1) * poss[evader_key][i][1]
      coords = torch.cat((coords, xcoords, ycoords), dim=1) 

      # X-Y coordinates of the evaders for the pairwise game
      pairwise_coords[evader_key] = torch.cat((time_coords, xcoords, ycoords, mgrid_coords), dim=1)

    # Setup the theta coordinates
    coords_ego_theta = ego_vehicle_theta[i] * torch.ones(mgrid_coords.shape[0], 1)/(math.pi * angle_alpha)
    coords = torch.cat((coords, coords_ego_theta), dim=1)
    for j in range(2):
      evader_key = '%i' %(j+1) + 'E'
      # Theta coordinates of the evaders for the full game
      tcoords = torch.ones(mgrid_coords.shape[0], 1) * thetas[evader_key][i]/(math.pi * angle_alpha)
      coords = torch.cat((coords, tcoords), dim=1)

      # Theta coordinates of the evaders for the pairwise game
      pairwise_coords[evader_key] = torch.cat((pairwise_coords[evader_key], tcoords, coords_ego_theta), dim=1)

    model_in = {'coords': coords[:, None, :].cuda()}
    model_out = model(model_in)

    # Detatch model ouput and reshape
    model_out = model_out['model_out'].detach().cpu().numpy()
    model_out = model_out.reshape((sidelen, sidelen))

    # Unnormalize the value function
    norm_to = 0.02
    mean = 0.25
    var = 0.5
    model_out = (model_out*var/norm_to) + mean 

    # Plot the zero level sets
    valfunc = model_out*1.
    model_out = (model_out <= level)*1.

    # Plot the actual data and small aircrafts
    ax = fig.add_subplot(1, num_slices, i+1)
    ax_valfunc = fig_valfunc.add_subplot(1, num_slices, i+1)
    ax_error = fig_error.add_subplot(1, num_slices, i+1)
    aircraft_size = 0.2
    sA = {}
    for j in range(2):
      evader_key = '%i' %(j+1) + 'E'
      aircraft_image = scipy.ndimage.rotate(plt.imread('resources/ego_aircraft.png'), 180.0*thetas[evader_key][i]/math.pi)
      sA[evader_key] = ax.imshow(aircraft_image, extent=(poss[evader_key][i][0]-aircraft_size, poss[evader_key][i][0]+aircraft_size, poss[evader_key][i][1]-aircraft_size, poss[evader_key][i][1]+aircraft_size))
      ax.plot(poss[evader_key][i][0], poss[evader_key][i][1], "o")
    s = ax.imshow(model_out.T, cmap='bwr_r', alpha=0.5, origin='lower', vmin=-1., vmax=1., extent=(-1., 1., -1., 1.))
    sV1 = ax_valfunc.imshow(valfunc.T, cmap='bwr_r', alpha=0.8, origin='lower', vmin=-0.2, vmax=0.2, extent=(-1., 1., -1., 1.))
    sV2 = ax_valfunc.contour(valfunc.T, cmap='bwr_r', alpha=0.5, origin='lower', vmin=-0.2, vmax=0.2, levels=30, extent=(-1., 1., -1., 1.))
    plt.clabel(sV2, levels=30, colors='k')
    fig_valfunc.colorbar(sV1) 

    # Compute and plot pairwise collision sets
    sP = {}
    model_out_pairwise_sofar = None
    valfunc_pairwise = None
    for j in range(2):
      evader_key = '%i' %(j+1) + 'E'
      model_in_pairwise = {'coords': pairwise_coords[evader_key].cuda()}
      model_out_pairwise = model_1P(model_in_pairwise)['model_out'].detach().cpu().numpy()
      model_out_pairwise = model_out_pairwise.reshape((sidelen, sidelen))
      norm_to_pairwise = 0.02
      mean_pairwise = 0.25
      var_pairwise = 0.5
      model_out_pairwise = (model_out_pairwise*var_pairwise/norm_to_pairwise) + mean_pairwise 

      if model_out_pairwise_sofar is None:
        model_out_pairwise_sofar = (model_out_pairwise <= level)*1.
        valfunc_pairwise = model_out_pairwise * 1.
      else:
        model_out_pairwise_sofar = np.clip((model_out_pairwise <= level)*1. + model_out_pairwise_sofar, 0., 1.)
        valfunc_pairwise = np.minimum(valfunc_pairwise, model_out_pairwise*1.0)

    s2 = ax.imshow(model_out_pairwise_sofar.T, cmap='seismic', alpha=0.5, origin='lower', vmin=-1., vmax=1., extent=(-1., 1., -1., 1.))

    # Error plot
    error = np.clip(model_out - model_out_pairwise_sofar, 0., 1.)
    ax_error.imshow(error.T, cmap='bwr', origin='lower', vmin=-1., vmax=1., extent=(-1., 1., -1., 1.))

    ## Append the value functions
    val_functions['pairwise'].append(valfunc_pairwise)
    val_functions['full'].append(valfunc)

  return fig, fig_error, fig_valfunc, val_functions

fig, fig_error, fig_valfunc, val_functions = val_fn_BRS_posspace(model, model_1P)
fig.savefig(os.path.join(logging_root, '2E1P_BRS_comparison.png'))
fig_valfunc.savefig(os.path.join(logging_root, '2E1P_BRS_raw.png'))
fig_error.savefig(os.path.join(logging_root, '2E1P_BRS_error.png'))
spio.savemat(os.path.join(logging_root, '2E1P_raw_valfuncs.mat'), val_functions)