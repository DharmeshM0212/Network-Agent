import numpy as np
import pandas as pd

num_samples = 15000
max_throughput = 20
time_step = 0.1

state_names = {0: "Poor", 1: "Moderate", 2: "Good"}

np.random.seed(42)

snr_ranges = {
    0: (0, 8),
    1: (8, 18),
    2: (18, 30)
}

# State-specific stay probabilities to balance classes
stay_probs = {0: 0.65, 1: 0.55, 2: 0.45}  # Poor stays a bit more, Good less
move_prob = 0.35  # Normal move
jump_prob = 0.20  # Occasional jump

time = np.arange(num_samples) * time_step
snr = np.zeros(num_samples)
ber = np.zeros(num_samples)
loss = np.zeros(num_samples)
jitter = np.zeros(num_samples)
throughput = np.zeros(num_samples)
labels = []

current_state = 1  # Start in moderate
snr[0] = np.random.uniform(*snr_ranges[current_state])
ber[0] = 0.5 * np.exp(-snr[0] / 3) + np.random.uniform(0, 0.002)
jitter[0] = np.random.normal(5, 2)
loss[0] = np.clip((ber[0] * 1000) + jitter[0] * 0.05, 0, 10)
throughput[0] = max_throughput * (1 - loss[0] / 100)
labels.append(state_names[current_state])

for i in range(1, num_samples):
    rnd = np.random.rand()
    state_stay_prob = stay_probs[current_state]

    if rnd < state_stay_prob:
        pass  # Stay in same state
    elif rnd < state_stay_prob + move_prob:
        # Move to adjacent state
        if np.random.rand() < 0.5:
            current_state = max(0, current_state - 1)
        else:
            current_state = min(2, current_state + 1)
    else:
        # Jump 2 states
        if np.random.rand() < 0.5:
            current_state = max(0, current_state - 2)
        else:
            current_state = min(2, current_state + 2)

    # Smooth SNR change
    prev_snr = snr[i-1]
    target_range = snr_ranges[current_state]
    target_snr = np.random.uniform(*target_range)
    snr[i] = np.clip(prev_snr + (target_snr - prev_snr) * np.random.uniform(0.1, 0.35),
                     *snr_ranges[current_state])
    snr[i] += np.random.normal(0, 0.3)

    # Compute features
    ber[i] = 0.5 * np.exp(-snr[i] / 3) + np.random.uniform(0, 0.002)
    jitter[i] = np.random.normal(5, 2)
    if snr[i] < 10:
        jitter[i] += np.random.normal(10, 5)

    loss[i] = np.clip((ber[i] * 1000) + jitter[i] * 0.05, 0, 10)
    throughput[i] = max_throughput * (1 - loss[i] / 100)
    labels.append(state_names[current_state])

df = pd.DataFrame({
    "Time": time,
    "SNR_dB": snr,
    "BER": ber,
    "PacketLoss_pct": loss,
    "Jitter_ms": jitter,
    "Throughput_Mbps": throughput,
    "Label": labels
})

df.to_csv("C:/Users/DHARMESH M/Documents/Projects/SIG-COG/Dataset/synthetic_link_dataset_balanced.csv", index=False)
print("Synthetic balanced dataset generated.")

print(df["Label"].value_counts())
