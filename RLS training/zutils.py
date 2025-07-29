from dataclasses import dataclass
import torch
import matplotlib.pyplot as plt

# @dataclass
# class TrialContext:
#     num_trials: int
#     trial_length: int
#     stim_length: int
#     memory_length: int
#     response_length: int
#     num_rules: int = 3
#     rule: int = 0  # Selected task index (0, 1, or 2)
#     freq_min: float = 1.0
#     freq_max: float = 10.0
#     device: str = 'cpu'
#     dtype: torch.dtype = torch.float

# def generate_snn_inputs(context: TrialContext):
#     FIX = 0
#     COS = 1
#     SIN = 2
#     RULE_START = 3

#     dt = 0.001  # 1 ms per timepoint

#     inputs = torch.zeros((context.num_trials, context.trial_length, 3 + context.num_rules),
#                          device=context.device, dtype=context.dtype)

#     context_end = context.trial_length - (context.stim_length + context.memory_length + context.response_length)
#     stim_start = context_end
#     stim_end = stim_start + context.stim_length
#     response_start = stim_end + context.memory_length

#     freqs = torch.empty(context.num_trials).uniform_(context.freq_min, context.freq_max)
#     context.freqs = freqs

#     inputs[:, :response_start, FIX] = 1.0

#     for i in range(context.num_trials):
#         t_stim = torch.arange(context.stim_length, device=context.device, dtype=context.dtype) * dt
#         stim_wave = torch.sin(2 * torch.pi * freqs[i] * t_stim)
#         inputs[i, stim_start:stim_end, SIN] = stim_wave
#         inputs[i, :stim_end, RULE_START + context.rule] = 1.0

#     return inputs

# def generate_snn_outputs(context: TrialContext):
#     FIX = 0
#     COS = 1
#     SIN = 2

#     dt = 0.001  # 1 ms per timepoint

#     outputs = torch.zeros((context.num_trials, context.trial_length, 3), device=context.device, dtype=context.dtype)

#     context_end = context.trial_length - (context.stim_length + context.memory_length + context.response_length)
#     stim_end = context_end + context.stim_length
#     response_start = stim_end + context.memory_length

#     outputs[:, :response_start, FIX] = 1.0

#     for i in range(context.num_trials):
#         t_resp = torch.arange(context.response_length, device=context.device, dtype=context.dtype) * dt
#         resp_wave = torch.sin(2 * torch.pi * context.freqs[i] * t_resp)
#         outputs[i, response_start:response_start + context.response_length, SIN] = resp_wave

#     return outputs

@dataclass
class TrialContext:
    num_trials: int
    trial_length: int
    stim_length: int
    memory_length: int
    response_length: int
    num_rules: int = 3
    rule: int = 0  # Selected task index (0, 1, or 2)
    freq_min: float = 1.0
    freq_max: float = 10.0
    time_step: float = 0.001  # seconds per timepoint
    device: str = 'cpu'
    dtype: torch.dtype = torch.float

def generate_snn_inputs(context: TrialContext):
    FIX = 0
    COS = 1
    SIN = 2
    RULE_START = 3

    inputs = torch.zeros((context.num_trials, context.trial_length, 3 + context.num_rules),
                         device=context.device, dtype=context.dtype)

    context_end = context.trial_length - (context.stim_length + context.memory_length + context.response_length)
    stim_start = context_end
    stim_end = stim_start + context.stim_length
    response_start = stim_end + context.memory_length

    freqs = torch.empty(context.num_trials).uniform_(context.freq_min, context.freq_max)
    context.freqs = freqs

    inputs[:, :response_start, FIX] = 1.0

    for i in range(context.num_trials):
        t_stim = torch.arange(context.stim_length, device=context.device, dtype=context.dtype) * context.time_step
        stim_wave = torch.sin(2 * torch.pi * freqs[i] * t_stim)
        inputs[i, stim_start:stim_end, SIN] = stim_wave
        inputs[i, :stim_end, RULE_START + context.rule] = 1.0
    
    inputs[:,:,FIX] = 1-inputs[:,:,FIX] 

    return inputs

def generate_snn_outputs(context: TrialContext):
    FIX = 0
    COS = 1
    SIN = 2

    outputs = torch.zeros((context.num_trials, context.trial_length, 3), device=context.device, dtype=context.dtype)

    context_end = context.trial_length - (context.stim_length + context.memory_length + context.response_length)
    stim_end = context_end + context.stim_length
    response_start = stim_end + context.memory_length

    outputs[:, :response_start, FIX] = 1.0

    for i in range(context.num_trials):
        t_resp = torch.arange(context.response_length, device=context.device, dtype=context.dtype) * context.time_step
        resp_wave = torch.sin(2 * torch.pi * context.freqs[i] * t_resp)
        outputs[i, response_start:response_start + context.response_length, SIN] = resp_wave

    outputs[:,:,FIX] = 1-outputs[:,:,FIX] 

    return outputs

def plot_trial(inputs, outputs, context, trial_idx=0):
    """
    Plots the input channels and desired outputs for a single trial,
    with shaded regions for Context, Stim, Memory, and Response phases.
    """
    # Phase boundaries
    context_end = context.trial_length - (context.stim_length + context.memory_length + context.response_length)
    stim_start = context_end
    stim_end = stim_start + context.stim_length
    memory_start = stim_end
    memory_end = memory_start + context.memory_length
    response_start = memory_end

    phase_regions = [
        (0, context_end, 'Context', '#e0e0e0'),
        (stim_start, stim_end, 'Stim', '#b3cde3'),
        (memory_start, memory_end, 'Memory', '#ccebc5'),
        (response_start, context.trial_length, 'Response', '#fbb4ae')
    ]

    time = range(inputs.shape[1])
    fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # Plot inputs
    for i in range(3):
        axs[0].plot(time, inputs[trial_idx, :, i].cpu().numpy(), label=f'Input {i}')
    axs[0].set_ylabel('Inputs')
    axs[0].legend(loc='upper right')

    # Plot outputs
    for i in range(outputs.shape[2]):
        axs[1].plot(time, outputs[trial_idx, :, i].cpu().numpy(), label=f'Output {i}')
    axs[1].set_ylabel('Outputs')
    axs[1].set_xlabel('Time Step')
    axs[1].legend(loc='upper right')

    # Shade phases
    for ax in axs:
        for start, end, label, color in phase_regions:
            ax.axvspan(start, end, color=color, alpha=0.3)
            ax.text((start+end)/2, ax.get_ylim()[1]*0.95, label, color='black',
                    ha='center', va='top', fontsize=10, alpha=0.7)

    plt.tight_layout()
    plt.show()