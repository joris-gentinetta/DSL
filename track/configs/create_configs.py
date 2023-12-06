import json

# Define the base template
base_template = {
    "data_config": {"n_workers": 1},
    "segmentation_config": {
        "n_workers": 40,
        "min_area": 50,
        "max_area": 200,
        "min_frontier": 0.0,
        "max_noise": 0.0,
        "threshold": 0.5
    },
    "linking_config": {
        "n_workers": 40,
        "distance_weight": 0.0,
        "max_neighbors": 5,
        "max_distance": 10
    },
    "tracking_config": {
        "appear_weight": -1,
        "disappear_weight": -1,
        "division_weight": -0.1,
        "dismiss_weight_guess": None,
        "include_weight_guess": None,
        "window_size": None,
        "overlap_size": 1,
        "solution_gap": 0.005,
        "time_limit": 3600,
        "method": 0,
        "n_threads": 40,
        "link_function": "power",
        "power": 4,
        "bias": -0.001
    }
}

# Define the segmentation channel assignments for each file
channel_assignments = {
    # All channels
    1: {'stardist': [0, 1, 2, 3], 'wssd': [0, 1, 2, 3], 'cellpose': [0, 1, 2, 3], 'wscp': [0, 1, 2, 3]},
    2: {'stardist': [0, 1, 2, 3], 'cellpose': [0, 1, 2, 3]},
    3: {'stardist': [0, 1, 2, 3], 'wssd': [0, 1, 2, 3]},
    4: {'cellpose': [0, 1, 2, 3], 'wscp': [0, 1, 2, 3]},
    # Channel 0
    11: {'stardist': [0], 'wssd': [0], 'cellpose': [0], 'wscp': [0]},
    12: {'stardist': [0], 'cellpose': [0]},
    13: {'stardist': [0], 'wssd': [0]},
    14: {'cellpose': [0], 'wscp': [0]},
    # Channel 1
    21: {'stardist': [1], 'wssd': [1], 'cellpose': [1], 'wscp': [1]},
    22: {'stardist': [1], 'cellpose': [1]},
    23: {'stardist': [1], 'wssd': [1]},
    24: {'cellpose': [1], 'wscp': [1]},
    # Channel 2
    31: {'stardist': [2], 'wssd': [2], 'cellpose': [2], 'wscp': [2]},
    32: {'stardist': [2], 'cellpose': [2]},
    33: {'stardist': [2], 'wssd': [2]},
    34: {'cellpose': [2], 'wscp': [2]},
    # Channel 3
    41: {'stardist': [3], 'wssd': [3], 'cellpose': [3], 'wscp': [3]},
    42: {'stardist': [3], 'cellpose': [3]},
    43: {'stardist': [3], 'wssd': [3]},
    44: {'cellpose': [3], 'wscp': [3]}
}

# Function to save the JSON file
def save_json(file_number, data):
    filename = f"{file_number}.json"
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)
    print(f"File {filename} saved.")

# Generate and save each file
for file_number, channels in channel_assignments.items():
    data = base_template.copy()
    data['segmentation_channels'] = channels
    save_json(file_number, data)
