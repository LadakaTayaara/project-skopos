import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.ensemble import RandomForestRegressor
import os
import time
import xml.etree.ElementTree as ET
from collections import defaultdict

def parse_acmi_xml(xml_file):
    print(f"Parsing ACMI XML file: {xml_file}")
    start_time = time.time()
    
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
    except Exception as e:
        print(f"Error parsing XML file: {e}")
        return None
    
    # Namespace handling (if present)
    ns = {'acmi': ''}  # Default empty namespace
    if '}' in root.tag:
        ns['acmi'] = root.tag.split('}')[0].strip('{')
    
    aerial_objects = {}
    total_objects = 0
    position_count = 0
    
    # Find all objects in the XML
    for obj in root.findall('.//acmi:Object', namespaces=ns) + root.findall('.//Object'):
        total_objects += 1
        obj_id = obj.get('ID', str(total_objects))
        obj_name = obj.get('Name', f'Object_{obj_id}')
        obj_type = obj.get('Type', 'Unknown')
        
        # Find all position records
        positions = []
        for record in obj.findall('.//acmi:Record', namespaces=ns) + obj.findall('.//Record'):
            # Try different position formats
            pos = None
            if record.get('X') and record.get('Y') and record.get('Z'):
                pos = [float(record.get('X')), float(record.get('Y')), float(record.get('Z'))]
            elif record.get('Lon') and record.get('Lat') and record.get('Alt'):
                # Simple conversion from Lat/Lon to X/Y (for demo purposes)
                lon = float(record.get('Lon'))
                lat = float(record.get('Lat'))
                alt = float(record.get('Alt'))
                pos = [lon * 100000, lat * 100000, alt]  # Crude approximation
            elif record.text and ',' in record.text:
                try:
                    pos = list(map(float, record.text.split(','))[:3])
                except Exception as e:
                    print(f"Error parsing position from record text: {record.text}, error: {e}")
                except:
                    pass
            
            if pos and len(pos) == 3:
                positions.append(pos)
        
        if len(positions) >= 10:
            position_count += len(positions)
            aerial_objects[obj_id] = {
                'name': f"{obj_name} ({obj_type})",
                'positions': np.array(positions, dtype=np.float32)
            }
    
    print(f"Parsing completed in {time.time() - start_time:.2f} seconds")
    print(f"Total objects found: {total_objects}")
    print(f"Valid aerial objects with position data: {len(aerial_objects)}")
    print(f"Total position points: {position_count}")
    
    if not aerial_objects:
        print("\nPossible reasons for no objects found:")
        print("1. Position data might be in different XML elements/attributes")
        print("2. The XML structure might differ from standard ACMI format")
        print("\nDebug info - first object example:")
        sample_obj = root.find('.//acmi:Object', namespaces=ns) or root.find('.//Object')
        if sample_obj is not None:
            print("Object attributes:", sample_obj.attrib)
            sample_record = sample_obj.find('.//acmi:Record', namespaces=ns) or sample_obj.find('.//Record')
            if sample_record is not None:
                print("Record attributes:", sample_record.attrib)
                print("Record text:", sample_record.text)
    
    return aerial_objects

def train_trajectory_model(positions, future_steps=15):
    if len(positions) < future_steps + 1:
        raise ValueError(f"Not enough positions ({len(positions)}) for training")
    
    print(f"Training model with {len(positions)} positions...")
    start_time = time.time()
    
    n_samples = len(positions) - future_steps - 1
    X = np.zeros((n_samples, 4))
    y = np.zeros((n_samples, future_steps * 3))
    
    for i in range(n_samples):
        time_feature = i / len(positions)
        X[i] = np.append(positions[i], time_feature)
        y[i] = positions[i+1:i+1+future_steps].flatten()
    
    model = RandomForestRegressor(
        n_estimators=min(100, 50 + len(positions) // 10),
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
            'b-', linewidth=2, label='Actual Path')
    ax.plot(predicted[:, 0], predicted[:, 1], predicted[:, 2], 
            'r--', linewidth=2, label='Predicted Path')
    
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Altitude (Z)')
    ax.set_title(f'Trajectory Prediction for {obj_name}')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.show()

def main():
    # Change this to your XML file path
    xml_file = r"C:\Users\Abhinandan Singh\Desktop\SCOPOS\demofile.xml"
    
    if not os.path.exists(xml_file):
        print(f"Error: File not found at {xml_file}")
        return
    
    aerial_objects = parse_acmi_xml(xml_file)
    
    if not aerial_objects:
        print("No valid objects found with position data.")
        return
    
    print("\nAvailable objects:")
    for i, (obj_id, obj) in enumerate(aerial_objects.items()):
        print(f"{i+1}. {obj['name']} (Points: {len(obj['positions'])})")
    
    while True:
        choice = input("\nSelect an object (number) or 'q' to quit: ")
        if choice.lower() == 'q':
            return
        
        try:
            choice = int(choice) - 1
            if 0 <= choice < len(aerial_objects):
                obj_id = list(aerial_objects.keys())[choice]
                obj = aerial_objects[obj_id]
                print(f"\nProcessing: {obj['name']}")
                
                try:
                    model, steps = train_trajectory_model(obj['positions'])
                    predicted = predict_trajectory(model, obj['positions'], steps)
                    plot_results(obj['positions'], predicted, obj['name'])
                except Exception as e:
                    print(f"Prediction error: {e}")
        except ValueError:
            print("Please enter a valid number or 'q' to quit")

if __name__ == "__main__":
    main()