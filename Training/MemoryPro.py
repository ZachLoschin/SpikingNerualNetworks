"""
Library import
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

import torch
import torch.nn as nn
import zutils


class SurrGradSpike(torch.autograd.Function):
    """
    Here we implement our spiking nonlinearity which also implements 
    the surrogate gradient. By subclassing torch.autograd.Function, 
    we will be able to use all of PyTorch's autograd functionality.
    Here we use the normalized negative part of a fast sigmoid 
    as this was done in Zenke & Ganguli (2018).
    """
    
    scale = 100.0 # controls steepness of surrogate gradient

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we compute a step function of the input Tensor
        and return it. ctx is a context object that we use to stash information which 
        we need to later backpropagate our error signals. To achieve this we use the 
        ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        out = torch.zeros_like(input)
        out[input > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor we need to compute the 
        surrogate gradient of the loss with respect to the input. 
        Here we use the normalized negative part of a fast sigmoid 
        as this was done in Zenke & Ganguli (2018).
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input/(SurrGradSpike.scale*torch.abs(input)+1.0)**2
        return grad
    
# here we overwrite our naive spike function by the "SurrGradSpike" nonlinearity which implements a surrogate gradient
spike_fn  = SurrGradSpike.apply

"""
Generate inputs and outputs for SNN training.
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # was: torch.device("cpu")

context = zutils.TrialContext(
    num_trials=256,
    trial_length=1200,
    stim_length=400,
    memory_length=10,
    response_length=600,
    num_rules=3,
    rule=0,
    freq_min=10.0,
    freq_max=50.0,
    time_step=1e-3,
    device=device,
    dtype=torch.float
)

inputs = zutils.generate_snn_inputs(context).to(device)
outputs = zutils.generate_snn_outputs(context).to(device)
# zutils.plot_trial(inputs, outputs, context, trial_idx=0)

# Network parameters
nb_inputs  = 6
nb_hidden  = 10
nb_outputs = 3
time_step = 1e-3

# Training parameters
nb_steps  = 1199
batch_size = 256
dtype = torch.float


# Setup the spiking network model
tau_mem = 10e-3  # Membrane voltage time constant
tau_syn = 5e-3  # Synaptic voltage time constant

# Decay constants for discretized equations
alpha   = float(np.exp(-time_step/tau_syn))
beta    = float(np.exp(-time_step/tau_mem))

# Initialize the weight matrices
weight_scale = 7*(1.0-beta) # this should give us some spikes to begin with

# Input to hidden layer
w1 = torch.empty((nb_inputs, nb_hidden),  device=device, dtype=dtype, requires_grad=True)
torch.nn.init.normal_(w1, mean=0.0, std=weight_scale/np.sqrt(nb_inputs))

# Hidden layer to output layer
w2 = torch.empty((nb_hidden, nb_outputs), device=device, dtype=dtype, requires_grad=True)
torch.nn.init.normal_(w2, mean=0.0, std=weight_scale/np.sqrt(nb_hidden))

# Recurrent connections
v = torch.empty((nb_hidden, nb_hidden), device=device, dtype=dtype, requires_grad=True)
torch.nn.init.normal_(v, mean=0.0, std=(weight_scale/np.sqrt(nb_hidden)))

h1 = torch.einsum("abc,cd->abd", (inputs, w1))

def run_snn(inputs):
    h1 = torch.einsum("abc,cd->abd", (inputs, w1))
    syn = torch.zeros((batch_size,nb_hidden), device=device, dtype=dtype)
    mem = torch.zeros((batch_size,nb_hidden), device=device, dtype=dtype)

    mem_rec = []
    spk_rec = []

    # Compute hidden layer activity
    for t in range(nb_steps):
        mthr = mem-1.0
        out = spike_fn(mthr)
        rst = out.detach() # We do not want to backprop through the reset

        new_syn = alpha*syn +h1[:,t]
        new_mem = (beta*mem +syn)*(1.0-rst)

        mem_rec.append(mem)
        spk_rec.append(out)
        
        mem = new_mem
        syn = new_syn

    mem_rec = torch.stack(mem_rec,dim=1)
    spk_rec = torch.stack(spk_rec,dim=1)

    # Readout layer
    h2= torch.einsum("abc,cd->abd", (spk_rec, w2))
    flt = torch.zeros((batch_size,nb_outputs), device=device, dtype=dtype)
    out = torch.zeros((batch_size,nb_outputs), device=device, dtype=dtype)
    out_rec = [out]
    for t in range(nb_steps):
        new_flt = alpha*flt +h2[:,t]
        new_out = beta*out +flt

        flt = new_flt
        out = new_out

        out_rec.append(out)

    out_rec = torch.stack(out_rec,dim=1)
    other_recs = [mem_rec, spk_rec]
    return out_rec, other_recs


params = [w1, w2, v]  # Parameters to optimize
optimizer = torch.optim.Adam(params, lr=2e-3, betas=(0.9, 0.999))  # Optimizer

loss_fn = nn.MSELoss()  # Mean squared error for regression

loss_hist = []

for e in range(5):
    print(e)
    # Run the network and get output
    o, _ = run_snn(inputs)  # output: (batch_size, timesteps, nb_outputs)
    
    # Compute the loss (compare output to desired outputs)
    # If outputs is (batch_size, timesteps, nb_outputs), use as is
    loss_val = loss_fn(o, outputs)
    
    # Update the weights
    optimizer.zero_grad()
    loss_val.backward()
    optimizer.step()
    
    # Store loss value
    loss_hist.append(loss_val.item())

loss_hist_true_grad = loss_hist  # Store for later use


plt.plot(loss_hist)
plt.xlabel("Epoch")
plt.ylabel("Loss")
sns.despine()
plt.savefig("loss_curve.png")  # Save the plot as a PNG file
plt.close()



# Run the trained model on the test data (or reuse training data for demonstration)
test_output, _ = run_snn(inputs)  # Shape: (batch_size, timesteps, nb_outputs)

# Select a few example trials to plot
def plot_trial_outputs(outputs, test_output, trial_idx, channel_names=None, save_path=None):
    """
    Plots desired vs actual outputs for all output channels for a given trial.
    If save_path is provided, saves the figure to that path.
    """
    num_channels = outputs.shape[2]
    if channel_names is None:
        channel_names = [f'Channel {i}' for i in range(num_channels)]
    plt.figure(figsize=(5 * num_channels, 4))
    for ch in range(num_channels):
        plt.subplot(1, num_channels, ch + 1)
        plt.plot(outputs[trial_idx, :, ch].cpu().numpy(), label='Desired', color='blue', linewidth=2)
        plt.plot(test_output[trial_idx, :, ch].detach().cpu().numpy(), label='Actual', color='red', linestyle='--')
        plt.title(f'Trial {trial_idx}: {channel_names[ch]}')
        plt.xlabel('Time Step')
        plt.ylabel('Amplitude')
        plt.legend()
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

# Example usage:
example_trials = [0, 1, 2]
channel_names = ['FIX', 'COS', 'SIN']  # Adjust if your output channels differ
for trial in example_trials:
    filename = f"trial_output_{trial}.png"
    plot_trial_outputs(outputs, test_output, trial, channel_names, save_path=filename)

# Save model parameters after training
save_dir = "saved_models"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "lif_model_params.pt")
torch.save({
    'w1': w1.detach().cpu(),
    'w2': w2.detach().cpu(),
    'v': v.detach().cpu()
}, save_path)
print(f"Model parameters saved to {save_path}")

# To load later:
# checkpoint = torch.load(save_path)
# w1 = checkpoint['w1'].to(device).requires_grad_()
# w2 = checkpoint['w2'].to(device).requires_grad_()