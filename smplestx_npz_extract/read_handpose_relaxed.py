import numpy as np

data = np.load("smplx_handposes.npz", allow_pickle=True)

print("keys:", data.files)

for k in data.files:
    print(f"\n{k}:")
    obj = data[k]
    print("type:", type(obj))
    print(obj)

hand_poses = data["hand_poses"].item()

print("\nhand_poses keys:", hand_poses.keys())

for preset_name, preset_value in hand_poses.items():
    print(f"\nPreset: {preset_name}")
    left_hand, right_hand = preset_value

    print(" left hand shape:", left_hand.shape)
    print(left_hand)

    print(" right hand shape:", right_hand.shape)
    print(right_hand)