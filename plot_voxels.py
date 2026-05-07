import json
import matplotlib.pyplot as plt
import numpy as np
import os

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
    
    # Get the bounding box of the points
    max_x, max_y, max_z = points.max(axis=0)
    
    # Create a 3D grid of False
    voxels = np.zeros((max_x + 1, max_y + 1, max_z + 1), dtype=bool)
    
    # Fill the grid with True where there are points
    for x, y, z in points:
        voxels[x, y, z] = True
        
    # Create a color array
    colors = np.empty(voxels.shape, dtype=object)
    
    # Color by Z-axis (height)
    cmap = plt.get_cmap('viridis')
    for x in range(max_x + 1):
        for y in range(max_y + 1):
            for z in range(max_z + 1):
                if voxels[x, y, z]:
                    norm_z = z / max_z if max_z > 0 else 0
                    rgba = cmap(norm_z)
                    colors[x, y, z] = rgba

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Draw voxels
    ax.voxels(voxels, facecolors=colors, edgecolor='k', linewidth=0.5, alpha=0.9)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Original Input Point Cloud (Voxelized)')
    
    # Better viewing angle
    ax.view_init(elev=30, azim=45)
    
    # Ensure axes ratio are equal
    ax.set_box_aspect((max_x, max_y, max_z))
    
    output_path = '/home/csgrad/cxu26/.gemini/antigravity/brain/5314eedb-b5e6-4644-93c9-80f6455e190b/artifacts/voxel_pc.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight')
    print("Saved to", output_path)
    
except Exception as e:
    import traceback
    traceback.print_exc()
