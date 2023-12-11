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
        "time_limit": 1200,
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
    2: {'stardist': [0, 1, 2, 3], 'wssd': [], 'cellpose': [0, 1, 2, 3], 'wscp': []},
    3: {'stardist': [0, 1, 2, 3], 'wssd': [0, 1, 2, 3], 'cellpose': [], 'wscp': []},
    4: {'stardist': [], 'wssd': [], 'cellpose': [0, 1, 2, 3], 'wscp': [0, 1, 2, 3]},
    5: {'stardist': [0, 1, 2, 3], 'wssd': [], 'cellpose': [], 'wscp': []},
    6: {'stardist': [], 'wssd': [], 'cellpose': [0, 1, 2, 3], 'wscp': []},
    # Channel 0
    11: {'stardist': [0], 'wssd': [0], 'cellpose': [0], 'wscp': [0]},
    12: {'stardist': [0], 'wssd': [], 'cellpose': [0], 'wscp': []},
    13: {'stardist': [0], 'wssd': [0], 'cellpose': [], 'wscp': []},
    14: {'stardist': [], 'wssd': [], 'cellpose': [0], 'wscp': [0]},
    15: {'stardist': [0], 'wssd': [], 'cellpose': [], 'wscp': []},
    16: {'stardist': [], 'wssd': [], 'cellpose': [0], 'wscp': []},
    # Channel 1
    21: {'stardist': [1], 'wssd': [1], 'cellpose': [1], 'wscp': [1]},
    22: {'stardist': [1], 'wssd': [], 'cellpose': [1], 'wscp': []},
    23: {'stardist': [1], 'wssd': [1], 'cellpose': [], 'wscp': []},
    24: {'stardist': [], 'wssd': [], 'cellpose': [1], 'wscp': [1]},
    25: {'stardist': [1], 'wssd': [], 'cellpose': [], 'wscp': []},
    26: {'stardist': [], 'wssd': [], 'cellpose': [1], 'wscp': []},
    # Channel 2
    31: {'stardist': [2], 'wssd': [2], 'cellpose': [2], 'wscp': [2]},
    32: {'stardist': [2], 'wssd': [], 'cellpose': [2], 'wscp': []},
    33: {'stardist': [2], 'wssd': [2], 'cellpose': [], 'wscp': []},
    34: {'stardist': [], 'wssd': [], 'cellpose': [2], 'wscp': [2]},
    35: {'stardist': [2], 'wssd': [], 'cellpose': [], 'wscp': []},
    36: {'stardist': [], 'wssd': [], 'cellpose': [2], 'wscp': []},
    # Channel 3
    41: {'stardist': [3], 'wssd': [3], 'cellpose': [3], 'wscp': [3]},
    42: {'stardist': [3], 'wssd': [], 'cellpose': [3], 'wscp': []},
    43: {'stardist': [3], 'wssd': [3], 'cellpose': [], 'wscp': []},
    44: {'stardist': [], 'wssd': [], 'cellpose': [3], 'wscp': [3]},
    45: {'stardist': [3], 'wssd': [], 'cellpose': [], 'wscp': []},
    46: {'stardist': [], 'wssd': [], 'cellpose': [3], 'wscp': []},
    # Channel 4
    51: {'stardist': [4], 'wssd': [4], 'cellpose': [4], 'wscp': [4]},
    52: {'stardist': [4], 'wssd': [], 'cellpose': [4], 'wscp': []},
    53: {'stardist': [4], 'wssd': [4], 'cellpose': [], 'wscp': []},
    54: {'stardist': [], 'wssd': [], 'cellpose': [4], 'wscp': [4]},
    55: {'stardist': [4], 'wssd': [], 'cellpose': [], 'wscp': []},
    56: {'stardist': [], 'wssd': [], 'cellpose': [4], 'wscp': []},

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
