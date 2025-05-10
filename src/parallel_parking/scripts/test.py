import cv2
import numpy as np
import matplotlib.pyplot as plt

def find_and_visualize_parking_slots(binary_map, gap_threshold):
    # Create a copy of the input map for visualization
    visualization = cv2.cvtColor(binary_map.copy(), cv2.COLOR_GRAY2BGR)
    visualization = np.zeros_like(visualization)
    
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

    return parking_slots, visualization

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

def find_largest_gap_in_contour(contour, binary_image):
    # Get bounding rectangle of the contour
    x, y, w, h = cv2.boundingRect(contour)
    
    # Create a mask for this contour
    mask = np.zeros_like(binary_image)
    cv2.drawContours(mask, [contour], 0, 255, -1)
    
    # Extract the region of interest
    roi = mask[y:y+h, x:x+w]
    
    # Create a distance transform of the inverse (gaps)
    # Distance transform assigns to each pixel the distance to the nearest non-zero pixel
    roi_inv = cv2.bitwise_not(roi)
    dist_transform = cv2.distanceTransform(roi_inv, cv2.DIST_L2, 3)
    
    # Find the maximum distance and its location
    max_dist = np.max(dist_transform)
    max_dist_loc = np.unravel_index(np.argmax(dist_transform), dist_transform.shape)
    
    # The maximum distance is the radius of the largest gap
    largest_gap_radius = max_dist
    
    # Get the center of the largest gap in the original image coordinates
    gap_center_y = y + max_dist_loc[0]
    gap_center_x = x + max_dist_loc[1]
    
    # Estimate depth: vertical extent of the connected component the max point belongs to
    ############
    # gap_center = (float(gap_center_x), float(gap_center_y))
    # # Find the minimum distance from the gap center to the contour
    # min_dist_to_contour = cv2.pointPolygonTest(contour, gap_center, True)
    # # The depth is the absolute value of this distance (since it's negative inside the contour)
    # depth = abs(min_dist_to_contour)
    ############## somehow works
    gap_center = (int(gap_center_x), int(gap_center_y))
    
    # Calculate the depth by finding the farthest point of the contour from the gap center
    # in the direction away from the contour's opening
    
    # First, compute the convex hull and find the defects
    hull = cv2.convexHull(contour, returnPoints=False)
    defects = cv2.convexityDefects(contour, hull)
    
    # If we have defects, use them to estimate the gap depth
    if defects is not None:
        # Find the defect closest to our gap center
        closest_defect_idx = -1
        min_distance = float('inf')
        
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            far_pt = tuple(contour[f][0])
            
            dist = np.sqrt((far_pt[0] - gap_center_x)**2 + (far_pt[1] - gap_center_y)**2)
            if dist < min_distance:
                min_distance = dist
                closest_defect_idx = i
        
        if closest_defect_idx != -1:
            # Get the defect points
            s, e, f, d = defects[closest_defect_idx, 0]
            start_pt = tuple(contour[s][0])
            end_pt = tuple(contour[e][0])
            far_pt = tuple(contour[f][0])
            
            # The depth is the distance from the farthest point to the line connecting start and end
            # This is approximately the depth of the gap
            # Convert depth from OpenCV format (1/256 units) to pixels
            depth = d / 256.0
            
            # Alternative depth calculation - perpendicular distance from far point to line
            # This might be more accurate in some cases
            num = abs((end_pt[1]-start_pt[1])*far_pt[0] - (end_pt[0]-start_pt[0])*far_pt[1] + 
                      end_pt[0]*start_pt[1] - end_pt[1]*start_pt[0])
            denom = np.sqrt((end_pt[1]-start_pt[1])**2 + (end_pt[0]-start_pt[0])**2)
            if denom != 0:
                perpendicular_depth = num / denom
                # Use the larger value as our depth
                depth = max(depth, perpendicular_depth)
    else:
        # Fallback to using twice the gap radius as depth if no defects are found
        # This assumes the gap extends roughly twice as deep as its width
        depth = largest_gap_radius * 2
        
    return {
        'radius': largest_gap_radius,
        'width': largest_gap_radius * 2,  # Diameter of the circle
        'center': (gap_center_x, gap_center_y),
        'depth': depth
    }


def visualize_largest_gap(binary_image, contour, gap_info):
    # Create a color visualization
    vis_image = cv2.cvtColor(binary_image.copy(), cv2.COLOR_GRAY2BGR)
    
    # Draw the contour
    cv2.drawContours(vis_image, [contour], 0, (0, 255, 0), 2)
    
    # Draw the largest gap as a circle
    cv2.circle(vis_image, gap_info['center'], int(gap_info['radius']), (0, 0, 255), 2)
    
    # Add text with gap width
    cv2.putText(vis_image, f"Gap width: {gap_info['width']:.1f}", 
                (gap_info['center'][0] - 50, gap_info['center'][1] - int(gap_info['radius']) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    return vis_image

def process_gap_info(gap_info, threshold=80, gap=40, dist = 40, num_waypoints=4):
    # this is not finished
    center = gap_info['center']
    width = gap_info['width'] 
    depth = gap_info['depth']
    slot_center = (center[0], center[1] - depth // 2)
    slot_center = (int(slot_center[0]), int(slot_center[1]))

    back_center = (int(center[0] + threshold), int(center[1] - depth // 2))
    # print(f"Final point: {slot_center}")

    front_center = (int(center[0] - width // 2), int(center[1] - depth // 2))

    front_out = (front_center[0], center[1] + gap)

    waypoints = []
    for i in range(num_waypoints):
        out = (front_out[0] - dist * (i + 1), front_out[1])
        waypoints.append(out)



    return slot_center, back_center, front_center, front_out, waypoints

map_path = "/home/yufeiyang/Documents/paralle-parking-roboracer/maps/agh/agh300a1_blacked.pgm"

thresh = 80 # distance from the car end to center
gap = 40 # distance from waypoint to the slot corner
dist = 40 # distance between waypoints at the front
num_waypoints = 4 # number of waypoints outside of the slot

image = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)
_, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
gap_threshold = 5  # pixel threshold for the internal gap's width or height
parking_slots = visualize_results(binary, gap_threshold)
gap_infos = []
for i in range(len(parking_slots)):
    largest_gap_info_i = find_largest_gap_in_contour(parking_slots[i], binary)
    print(largest_gap_info_i)
    slot_center, back_center, front_center, front_out, waypoints = process_gap_info(largest_gap_info_i, thresh, gap, dist, num_waypoints)
    # vis = visualize_largest_gap(binary, parking_slots[i], largest_gap_info_i)
    visualization = visualize_largest_gap(binary, parking_slots[0], largest_gap_info_i)
    cv2.circle(visualization, slot_center, radius=7, color=(0, 0, 255), thickness=-1)
    cv2.circle(visualization, back_center, radius=7, color=(0, 0, 255), thickness=-1)
    cv2.circle(visualization, front_center, radius=7, color=(0, 0, 255), thickness=-1)
    cv2.circle(visualization, front_out, radius=7, color=(0, 0, 255), thickness=-1)
    cv2.circle(visualization, waypoints[0], radius=7, color=(0, 0, 255), thickness=-1)
    cv2.circle(visualization, waypoints[1], radius=7, color=(0, 0, 255), thickness=-1)
    cv2.circle(visualization, waypoints[2], radius=7, color=(0, 0, 255), thickness=-1)
    plt.figure(figsize=(6, 6))
    plt.title('Largest Gap Visualization')
    plt.imshow(cv2.cvtColor(visualization, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
    gap_infos.append(largest_gap_info_i)

all_points = np.vstack([back_center, slot_center, front_center, front_out] + waypoints)

print(all_points)

np.savetxt("waypoints.txt", all_points, delimiter=",", fmt='%d')
# print(gap_infos)

# visualization = visualize_largest_gap(binary, parking_slots[0], largest_gap_info)
# plt.figure(figsize=(6, 6))
# plt.title('Largest Gap Visualization')
# plt.imshow(cv2.cvtColor(visualization, cv2.COLOR_BGR2RGB))
# plt.axis('off')
# plt.show()
# print(largest_gap_info)