import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os

# read the data
try:
    with open('datasets_pc/test.jsonl', 'r') as f:
        msg = json.loads(f.readline())['messages'][1]['content']
        pc_str = msg.split('### Input Point Cloud:\n')[1].strip()
        points = []
        for p in pc_str.split('), ('):
            p = p.replace('(', '').replace(')', '')
            x, y, z = map(int, p.split(','))
            points.append((x, y, z))
            
    points = np.array(points)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=points[:, 2], cmap='viridis', marker='o', s=50, alpha=0.8)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Original Input Point Cloud')
    ax.view_init(elev=30, azim=45)
    
    output_path = '/home/csgrad/cxu26/.gemini/antigravity/brain/5314eedb-b5e6-4644-93c9-80f6455e190b/artifacts/original_pc.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print("Saved to", output_path)
    
except Exception as e:
    print("Error:", e)
