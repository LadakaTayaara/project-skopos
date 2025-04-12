import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pyacmi
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import os
import time

def load_and_filter_acmi(file_path):
    print("Loading ACMI file...")
    start_time = time.time()
    
    try:
        acmi = pyacmi.Acmi(file_path)
    except Exception as e:
        print(f"Error loading ACMI file: {e}")
        return None
    
    aerial_objects = {}
    total_objects = len(acmi.objects)
    print(f"Processing {total_objects} objects...")
    
    for i, obj in enumerate(acmi.objects):
        if i % 100 == 0:
            print(f"Processed {i}/{total_objects} objects...")
        
        if not hasattr(obj, 'position') or len(obj.position) < 10:
            continue
            
        positions = []
        try:
            for pos in obj.position:
                if hasattr(pos, 'x') and hasattr(pos, 'y') and hasattr(pos, 'z'):
                    positions.append([pos.x, pos.y, pos.z])
        except Exception as e:
            print(f"Error processing positions for object {obj.id}: {e}")
            continue
        
        if len(positions) > 20:
            obj_name = getattr(obj, 'name', f'Object_{obj.id}')
            obj_type = getattr(obj, 'type', 'Unknown')
            aerial_objects[obj.id] = {
                'name': f"{obj_name} ({obj_type})",
                'positions': np.array(positions, dtype=np.float32)
            }
    
    print(f"ACMI loading completed in {time.time() - start_time:.2f} seconds")
    print(f"Found {len(aerial_objects)} valid aerial objects")
    return aerial_objects

def train_trajectory_model(positions, future_steps=15):
    start_time = time.time()
    print(f"Training model with {len(positions)} positions...")
    
    n_samples = len(positions) - future_steps - 1
    X = np.zeros((n_samples, 4))
    y = np.zeros((n_samples, future_steps * 3))
    
    for i in range(n_samples):
        time_feature = i / len(positions)
        X[i] = np.append(positions[i], time_feature)
        y[i] = positions[i+1:i+1+future_steps].flatten()
    
    n_estimators = min(100, 50 + len(positions) // 10)
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=42,
        n_jobs=-1,
        max_depth=10,
        min_samples_split=5
    )
    
    model.fit(X, y)
    print(f"Model trained in {time.time() - start_time:.2f} seconds")
    return model, future_steps

def predict_trajectory(model, history, future_steps):
    predictions = np.zeros((future_steps, 3))
    last_known = history[-1]
    
    for step in range(future_steps):
        time_feature = (len(history) + step) / (len(history) + future_steps)
        features = np.append(last_known, time_feature)
        next_pos = model.predict([features])[0][:3]
        predictions[step] = next_pos
        last_known = next_pos
    
    return predictions

def plot_results(history, predicted, obj_name):
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot(history[:, 0], history[:, 1], history[:, 2], 
            'b-', linewidth=3, label='Actual Path', alpha=0.8)
    
    ax.plot(predicted[:, 0], predicted[:, 1], predicted[:, 2], 
            'r--', linewidth=3, label='Predicted Path', alpha=0.8)
    
    ax.plot([history[-1, 0], predicted[0, 0]], 
            [history[-1, 1], predicted[0, 1]], 
            [history[-1, 2], predicted[0, 2]], 
            'g-', linewidth=2, label='Prediction Start')
    
    ax.set_xlabel('X Coordinate', fontsize=12)
    ax.set_ylabel('Y Coordinate', fontsize=12)
    ax.set_zlabel('Altitude (Z)', fontsize=12)
    ax.set_title(f'Trajectory Prediction for {obj_name}', fontsize=14, pad=20)
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    plt.show()

def main():
    acmi_file = r"C:\Users\Abhinandan Singh\Desktop\presentation\demofile.acmi"
    
    if not os.path.exists(acmi_file):
        print(f"Error: File not found at {acmi_file}")
        print("Please verify the path and try again.")
        return
    
    print(f"Processing ACMI file: {os.path.basename(acmi_file)}")
    aerial_objects = load_and_filter_acmi(acmi_file)
    
    if not aerial_objects:
        print("No valid aerial objects found in the ACMI file.")
        print("Possible reasons:")
        print("- The file format might not be supported")
        print("- No objects with sufficient trajectory data")
        print("- All objects might be below altitude threshold")
        return
    
    print("\nAerial objects available for prediction:")
    for i, (obj_id, obj) in enumerate(aerial_objects.items()):
        print(f"{i+1}. {obj['name']} (Points: {len(obj['positions'])})")
    
    while True:
        try:
            choice = input("\nSelect an aircraft to predict (number) or 'q' to quit: ")
            if choice.lower() == 'q':
                return
            choice = int(choice) - 1
            if 0 <= choice < len(aerial_objects):
                break
            print(f"Please enter a number between 1 and {len(aerial_objects)}")
        except ValueError:
            print("Invalid input. Please enter a number.")
    
    obj_id = list(aerial_objects.keys())[choice]
    obj = aerial_objects[obj_id]
    print(f"\nAnalyzing: {obj['name']}")
    
    model, future_steps = train_trajectory_model(obj['positions'])
    predicted = predict_trajectory(model, obj['positions'], future_steps)
    
    plot_results(obj['positions'], predicted, obj['name'])

if __name__ == "__main__":
    main()
