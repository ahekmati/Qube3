import pickle
import os
import json

ma_cache_path = 'ma_cache'
combined_dict = {}

for filename in os.listdir(ma_cache_path):
    if filename.endswith('.pkl'):
        file_path = os.path.join(ma_cache_path, filename)
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        combined_dict[filename] = data

# Attempt to export to JSON
with open('combined_ma_cache.json', 'w') as json_file:
    json.dump(combined_dict, json_file, indent=4, default=str)  # 'default=str' helps with most non-serializable objects
print("Combined MA cache saved to combined_ma_cache.json")