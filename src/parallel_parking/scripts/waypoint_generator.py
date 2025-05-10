import cv2
import numpy as np
import matplotlib.pyplot as plt

def find_and_visualize_parking_slots(binary_map, gap_threshold):
    # Create a copy of the input map for visualization
    visualization = cv2.cvtColor(binary_map.copy(), cv2.COLOR_GRAY2BGR)
    
    # Find all contours in the binary map
    contours, hierarchy = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    parking_slots = []
    
    for contour in contours:
        # Get bounding rectangle of the contour
        x, y, w, h = cv2.boundingRect(contour)
        
        # Create a mask for this contour region
        region_mask = np.zeros_like(binary_map)
        cv2.drawContours(region_mask, [contour], 0, 255, -1)
        
        # Extract the region
        region = binary_map[y:y+h, x:x+w] & region_mask[y:y+h, x:x+w]
        
        # Analyze the gaps as before
        has_wide_gap = False
        
        # Horizontal scan (for vertical gaps)
        for row in range(h):
            current_gap_width = 0
            for col in range(w):
                if region[row, col] == 0:  # Empty space (gap)
                    current_gap_width += 1
                else:
                    if current_gap_width > gap_threshold:
                        has_wide_gap = True
                        break
                    current_gap_width = 0
            
            if current_gap_width > gap_threshold:  # Check if gap extends to edge
                has_wide_gap = True
            
            if has_wide_gap:
                break
        
        # Vertical scan (for horizontal gaps) if no wide gaps found horizontally
        if not has_wide_gap:
            for col in range(w):
                current_gap_width = 0
                for row in range(h):
                    if region[row, col] == 0:  # Empty space (gap)
                        current_gap_width += 1
                    else:
                        if current_gap_width > gap_threshold:
                            has_wide_gap = True
                            break
                        current_gap_width = 0
                
                if current_gap_width > gap_threshold:  # Check if gap extends to edge
                    has_wide_gap = True
                
                if has_wide_gap:
                    break
        
        if has_wide_gap:
            parking_slots.append(contour)
            # Draw this contour in green to indicate a parking slot
            cv2.drawContours(visualization, [contour], 0, (0, 255, 0), 2)
        else:
            # Draw other contours in red
            cv2.drawContours(visualization, [contour], 0, (0, 0, 255), 1)
    
    return parking_slots, visualization

# Example usage:
def visualize_results(binary_map, gap_threshold):
    # Make sure the binary map is properly formatted (uint8 type, 0 and 255 values)
    if binary_map.dtype != np.uint8:
        binary_map = binary_map.astype(np.uint8) * 255
    
    parking_slots, visualization = find_and_visualize_parking_slots(binary_map, gap_threshold)
    
    # Display results
    plt.figure(figsize=(12, 6))
    
    plt.subplot(121)
    plt.title('Original Binary Map')
    plt.imshow(binary_map, cmap='gray')
    plt.axis('off')
    
    plt.subplot(122)
    plt.title(f'Detected Parking Slots (Gap > {gap_threshold})')
    plt.imshow(cv2.cvtColor(visualization, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print(f"Found {len(parking_slots)} parking slots with gaps wider than {gap_threshold} pixels.")
    
    return parking_slots

map_path = "/home/yufeiyang/Documents/paralle-parking-roboracer/src/f1tenth_gym_ros/maps/park1.png"

image = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)
_, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
gap_threshold = 300  # pixel threshold for the internal gap's width or height
parking_slots = visualize_results(binary, gap_threshold)