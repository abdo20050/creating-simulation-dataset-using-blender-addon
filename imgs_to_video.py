import cv2
import os
import csv
import re
import math
from natsort import natsorted
from collections import defaultdict

def load_csv_data(csv_path):
    """
    Reads the CSV file and returns a dictionary mapping frame numbers to list of point data.
    Format: { frame_number: [{'name': str, 'x': int, 'y': int}, ...], ... }
    """
    points_map = defaultdict(list)
    
    if not os.path.exists(csv_path):
        print(f"Warning: CSV file not found at {csv_path}. Annotated videos will have no points.")
        return points_map
        
    print(f"Loading points from {csv_path}...")
    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    frame = int(row['Frame'])
                    x = int(row['X'])
                    y = int(row['Y'])
                    # Store Object Name if available, otherwise default to "Unknown"
                    obj_name = row.get('Object Name', 'Unknown')
                    
                    points_map[frame].append({
                        'name': obj_name,
                        'x': x, 
                        'y': y
                    })
                except (ValueError, KeyError):
                    continue
    except Exception as e:
        print(f"Error reading CSV: {e}")
        
    return points_map

def get_next_video_index(output_prefix):
    """
    Scans the output folder to find the next available video index 
    based on existing '_clean.mp4' files.
    """
    folder = os.path.dirname(output_prefix)
    prefix = os.path.basename(output_prefix)
    
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)

    # Look for files like "output_1_clean.mp4"
    existing = [f for f in os.listdir(folder) if f.startswith(prefix) and "_clean.mp4" in f]
    
    indices = []
    for filename in existing:
        # Extract the index number between the prefix and '_clean'
        # Regex explanation: prefix + "_" + (digits) + "_clean.mp4"
        match = re.search(rf"{re.escape(prefix)}_(\d+)_clean\.mp4", filename)
        if match:
            indices.append(int(match.group(1)))
    
    return max(indices, default=0) + 1

def split_images_to_videos(input_folder, output_prefix, csv_path, fps=24, frames_per_video=100, frame_size=None):
    # 1. Load the Points Data
    points_map = load_csv_data(csv_path)

    # 2. Get and sort image files
    images = [img for img in os.listdir(input_folder) if img.lower().endswith(('.jpg', '.jpeg', '.png'))]
    images = natsorted(images)

    total_images = len(images)
    if total_images == 0:
        raise ValueError("No images found in the folder.")

    # 3. Setup Frame Size from first image
    first_frame_path = os.path.join(input_folder, images[0])
    first_frame = cv2.imread(first_frame_path)
    if first_frame is None:
        raise RuntimeError(f"Could not read {first_frame_path}")
    
    if frame_size is None:
        height, width = first_frame.shape[:2]
        frame_size = (width, height)

    # 4. Calculation
    total_videos = math.ceil(total_images / frames_per_video) # changed floor to ceil to include remainder
    starting_index = get_next_video_index(output_prefix)

    print(f"Total frames: {total_images}")
    print(f"Frames per video: {frames_per_video}")
    print(f"Total video pairs to create: {total_videos}")
    print(f"Starting index: {starting_index}")
    
    # Codec setup
    fourcc = cv2.VideoWriter_fourcc(*'avc1') # Or 'mp4v' if X265 fails

    for i in range(total_videos):
        start_idx = i * frames_per_video
        end_idx = min(start_idx + frames_per_video, total_images)
        video_images = images[start_idx:end_idx]

        current_vid_idx = starting_index + i
        
        # Define output paths
        clean_path = f"{output_prefix}_{current_vid_idx}_clean.mp4"
        annotated_path = f"{output_prefix}_{current_vid_idx}_annotated.mp4"
        video_csv_path = f"{output_prefix}_{current_vid_idx}.csv"

        out_clean = cv2.VideoWriter(clean_path, fourcc, fps, frame_size)
        out_annotated = cv2.VideoWriter(annotated_path, fourcc, fps, frame_size)
        
        # Prepare list to collect CSV data for this specific video chunk
        chunk_csv_data = []
        
        print(f"Processing chunk {i+1}/{total_videos}...")

        # Enumerate to get local_frame_index (0 to frames_per_video-1)
        for local_frame_index, img_name in enumerate(video_images):
            img_path = os.path.join(input_folder, img_name)
            frame = cv2.imread(img_path)
            
            if frame is None:
                print(f"Warning: Skipping {img_path}")
                continue
                
            frame = cv2.resize(frame, frame_size)
            
            # --- 1. Write Clean Frame ---
            out_clean.write(frame)
            
            # --- 2. Process Data & Write Annotated Frame ---
            # Extract global frame number from filename (e.g., "render_frame_0042.jpg" -> 42)
            nums = re.findall(r'\d+', img_name)
            if nums:
                global_frame_num = int(nums[-1])
                points = points_map.get(global_frame_num, [])
                
                # Add points to the chunk CSV data
                for p in points:
                    chunk_csv_data.append({
                        'Frame': local_frame_index, # relative to start of this video (0-based)
                        'Object Name': p['name'],
                        'X': p['x'],
                        'Y': p['y']
                    })
                
                # Draw red circles for annotated video
                for p in points:
                    cv2.circle(frame, (p['x'], p['y']), 4, (0, 0, 255), -1)
            
            out_annotated.write(frame)
        
        out_clean.release()
        out_annotated.release()
        
        # --- 3. Save Chunk CSV ---
        if chunk_csv_data:
            try:
                with open(video_csv_path, 'w', newline='') as f_csv:
                    writer = csv.DictWriter(f_csv, fieldnames=['Frame', 'Object Name', 'X', 'Y'])
                    writer.writeheader()
                    writer.writerows(chunk_csv_data)
                print(f"Saved CSV: {video_csv_path}")
            except Exception as e:
                print(f"Error saving CSV {video_csv_path}: {e}")
        else:
            print(f"Warning: No point data found for {video_csv_path}, skipping CSV creation.")

        print(f"Saved Video: {clean_path}")

if __name__ == "__main__":
    # --- UPDATE PATHS HERE ---
    INPUT_FOLDER = r"C:/Users/abdulna/OneDrive - KAUST/simulation_pics/img_output"
    OUTPUT_PREFIX = r"C:/Users/abdulna/OneDrive - KAUST/simulation_pics/vid_output/output"
    
    # Path to the CSV generated by the Blender script
    CSV_PATH = os.path.join(INPUT_FOLDER, "pixel_coords_all_frames.csv")
    
    split_images_to_videos(
        input_folder=INPUT_FOLDER,
        output_prefix=OUTPUT_PREFIX,
        csv_path=CSV_PATH,
        fps=24,
        frames_per_video=120
    )