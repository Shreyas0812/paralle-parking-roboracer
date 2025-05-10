import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv
def calculate_gap_depth(binary_map, entrance_box, direction, mid_point, width, max_depth=100):
    """
    Calculate the depth of a gap by tracing along the direction vector.
    
    Args:
        binary_map: Binary image where white (255) represents free space
        entrance_box: The four corners of the entrance box
        direction: Unit vector pointing inward from the entrance
        mid_point: Midpoint of the entrance
        width: Width of the entrance
        max_depth: Maximum depth to check
    
    Returns:
        depth: The depth of the gap
        opens_up: Boolean indicating if the gap opens into a larger space
    """
    h, w = binary_map.shape
    current_point = mid_point.copy()
    depth = 0
    opens_up = False
    
    # Trace along the direction vector until we hit an obstacle or reach max depth
    for step in range(1, max_depth + 1):
        # Calculate the next point along the direction
        next_point = mid_point + (step * direction).astype(int)
        
        # Check if we're still within image bounds
        if not (0 <= next_point[0] < w and 0 <= next_point[1] < h):
            break
        
        # Check if we've hit an obstacle (black pixel)
        if binary_map[next_point[1], next_point[0]] == 0:
            depth = step - 1  # The last valid step
            break
        
        current_point = next_point
        depth = step
        
        # Check for widening of the gap (optional)
        # We can check perpendicular to the direction to see if the gap widens
        if step > width / 2:  # Only check after we've gone in a bit
            # Check width at this depth by scanning perpendicular to direction
            perpendicular = np.array([-direction[1], direction[0]])
            local_width = 0
            
            # Scan in both perpendicular directions
            for mult in [-1, 1]:
                for offset in range(1, int(width * 1.5)):  # Check up to 1.5x the entrance width
                    test_point = current_point + (mult * offset * perpendicular).astype(int)
                    
                    # Check if we're still within image bounds and on free space
                    if not (0 <= test_point[0] < w and 0 <= test_point[1] < h):
                        break
                    
                    if binary_map[test_point[1], test_point[0]] == 0:
                        break
                    
                    local_width += 1
            
            # If the width at this depth is significantly larger than the entrance width,
            # we can consider that the gap "opens up" into a larger space
            if local_width > width * 1.2:
                opens_up = True
                break
    
    return depth, opens_up

def detect_entrance_gaps(binary_map, gap_threshold, upper_threshold, debug=False):
    """
    Detect entrance gaps in a binary map contour.
    
    Args:
        binary_map: Binary image where white (255) represents free space
        gap_threshold: Minimum width (in pixels) to consider a gap
        debug: If True, displays intermediate processing steps
    
    Returns:
        list of gaps detected, visualization image
    """
    # Find contours in the binary map
    contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("No contours found in the image.")
        return [], cv2.cvtColor(binary_map, cv2.COLOR_GRAY2BGR)
    
    # Find the largest contour (the room boundary)
    main_contour = max(contours, key=cv2.contourArea)
    
    # Create visualization image
    vis = cv2.cvtColor(binary_map, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(vis, [main_contour], 0, (0, 0, 255), 2)
    
    # Approximate the polygon to simplify the contour
    epsilon = 0.01 * cv2.arcLength(main_contour, True)
    approx_contour = cv2.approxPolyDP(main_contour, epsilon, True)
    
    # Track detected gaps
    detected_gaps = []
    
    # Analyze the contour segments to find gaps
    for i in range(len(approx_contour)):
        p1 = approx_contour[i][0]
        p2 = approx_contour[(i + 1) % len(approx_contour)][0]
        
        # Calculate the segment length
        segment_length = np.linalg.norm(p2 - p1)
        
        # Skip very short segments
        if segment_length < gap_threshold:
            continue

        if segment_length > upper_threshold:
            # Skip long segments (could be walls)
            continue
        
        # Calculate segment midpoint and direction
        mid_point = ((p1 + p2) / 2).astype(int)
        direction = p2 - p1
        
        # Normalize direction vector
        direction_norm = direction / np.linalg.norm(direction)
        
        # Calculate perpendicular direction (pointing inward)
        perp_direction = np.array([-direction_norm[1], direction_norm[0]])
        
        # Check both possible perpendicular directions
        for multiplier in [-1, 1]:
            # Create a test point in perpendicular direction
            test_direction = multiplier * perp_direction
            test_point = mid_point + (10 * test_direction).astype(int)
            
            # Make sure the test point is within image bounds
            h, w = binary_map.shape
            if 0 <= test_point[0] < w and 0 <= test_point[1] < h:
                # If the test point is on white (inside the contour), this is the inward direction
                if binary_map[test_point[1], test_point[0]] == 255:
                    perp_direction = multiplier * perp_direction
                    break
        
        # Create points to check if this segment might be a gap entrance
        # Move slightly inward from the edge
        inner_mid_point = mid_point + (5 * perp_direction).astype(int)
        
        # Calculate a point further inside to check if this is truly an entrance
        deep_inner_point = mid_point + (20 * perp_direction).astype(int)
        
        # Ensure points are within image bounds
        h, w = binary_map.shape
        if (0 <= inner_mid_point[0] < w and 0 <= inner_mid_point[1] < h and
            0 <= deep_inner_point[0] < w and 0 <= deep_inner_point[1] < h):
            
            # Check if both points are on white (free space) - this likely indicates an entrance
            if (binary_map[inner_mid_point[1], inner_mid_point[0]] == 255 and
                binary_map[deep_inner_point[1], deep_inner_point[0]] == 255):
                
                # Calculate entrance width (perpendicular to the entrance direction)
                # by tracing along the contour segment
                width_direction = direction_norm
                width = segment_length
                
                # This seems to be an entrance and exceeds our threshold
                if width >= gap_threshold and width <= upper_threshold:
                    # Calculate the four corners of the entrance box
                    entrance_depth = 20  # How far the entrance box extends inside
                    
                    # Calculate entrance box corners
                    corner1 = p1
                    corner2 = p2
                    corner3 = p2 + (entrance_depth * perp_direction).astype(int)
                    corner4 = p1 + (entrance_depth * perp_direction).astype(int)
                    
                    entrance_box = np.array([corner1, corner2, corner3, corner4])
                    depth, found_opening = calculate_gap_depth(
                        binary_map, 
                        entrance_box, 
                        perp_direction, 
                        mid_point, 
                        width
                    )                    

                    # Store the detected gap
                    detected_gaps.append({
                        'box': entrance_box,
                        'width': width,
                        'midpoint': mid_point,
                        'direction': perp_direction,
                        'depth': depth,
                        'opens_up': found_opening  # The deepest point still in free space
                    })

                    # # Draw the entrance box
                    # cv2.polylines(vis, [entrance_box.reshape((-1, 1, 2))], True, (0, 255, 0), 2)
                    
                    # # Draw the direction vector
                    # cv2.arrowedLine(vis, tuple(mid_point), 
                    #                tuple((mid_point + 15 * perp_direction).astype(int)),
                    #                (255, 0, 255), 2)
    
    if debug:
        plt.figure(figsize=(12, 8))
        plt.subplot(121)
        plt.title("Original Binary Map")
        plt.imshow(binary_map, cmap='gray')
        
        plt.subplot(122)
        plt.title(f"Detected Gaps (min width: {gap_threshold}px)")
        plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        plt.tight_layout()
        plt.show()
    
    return detected_gaps, vis

def visualize_gaps(binary_map, gap_threshold, upper_threshold):
    """
    Detect and visualize gaps in a binary map.
    
    Args:
        binary_map: Binary image where white (255) represents free space
        gap_threshold: Minimum width to consider a gap
    """
    # Ensure the binary map is properly thresholded
    if len(binary_map.shape) > 2:
        binary_map = cv2.cvtColor(binary_map, cv2.COLOR_BGR2GRAY)
    
    _, binary_map = cv2.threshold(binary_map, 127, 255, cv2.THRESH_BINARY)
    
    # Detect gaps
    gaps, vis = detect_entrance_gaps(binary_map, gap_threshold, upper_threshold, debug=False)

    # visualize the gaps
    # for gap in gaps:
    gap = gaps[1]
    # Draw the depth line
    depth_end = gap['midpoint'] + (gap['depth'] * gap['direction']).astype(int)
    # cv2.line(vis, tuple(gap['midpoint']), tuple(depth_end), (255, 255, 0), 2)
    
    # Annotate with depth information
    text_pos = ((gap['midpoint'] + depth_end) // 2).astype(int)
    # cv2.putText(vis, f"{gap['depth']:.1f}px", tuple(text_pos), 
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    shifted_point1, shifted_point2, center_point, outer_waypoint1, outer_waypoint2, outer_waypoint3, mid = find_waypoint(gap)
    # save these points in a csv file


    cv2.circle(vis, tuple(center_point), 1, (0, 0, 255), -1)
    cv2.circle(vis, tuple(shifted_point1), 1, (0, 255, 255), -1)
    cv2.circle(vis, tuple(shifted_point2), 1, (0, 0, 255), -1)
    cv2.circle(vis, tuple(outer_waypoint1), 1, (0, 0, 255), -1)
    cv2.circle(vis, tuple(outer_waypoint2), 1, (0, 0, 255), -1)
    cv2.circle(vis, tuple(outer_waypoint3), 1, (0, 0, 255), -1)
    cv2.circle(vis, tuple(mid), 1, (0, 0, 255), -1)

    # Visualize results
    plt.figure(figsize=(12, 6))
    
    plt.subplot(121)
    plt.title("Original Map")
    plt.imshow(binary_map, cmap='gray')
    plt.axis('off')
    
    plt.subplot(122)
    plt.title(f"Detected Gaps (width > {gap_threshold}px)")
    plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print(f"Detected {len(gaps)} gaps with width > {gap_threshold}px")
    return gaps

def find_waypoint(gap_info, center_offset=6, gap_offset=6, angle_offset=np.pi/6):
    
    depth = gap_info['depth']
    midpoint = gap_info['midpoint']
    direction = gap_info['direction']
    box = gap_info['box']
    # center point of the gap
    center_point = midpoint + (depth * direction / 2).astype(int)
    # find the direction perpendicular to the gap direction
    perpendicular_direction = np.array([-direction[1], direction[0]])

    # compute the angle between perpendicular direction and the horizontal axis
    angle = np.arctan2(perpendicular_direction[1], perpendicular_direction[0])
    print(f"Angle: {angle} radians")
    # point alinged with the center point on the perpendicular direction but shifted
    shifted_point1 = center_point + (center_offset * perpendicular_direction).astype(int)
    shifted_point2 = center_point - (center_offset * perpendicular_direction).astype(int)
    # right lower corner of the gap
    corner1 = box[0] + (depth * direction).astype(int)
    outer_waypoint1 = corner1 + (gap_offset * direction).astype(int)
    outer_waypoint2 = outer_waypoint1 - (center_offset * perpendicular_direction).astype(int)
    outer_waypoint3 = outer_waypoint2 - (center_offset * perpendicular_direction).astype(int)
    # mid = ((corner1 + outer_waypoint1) / 2).astype(int)
    # modify mid such that it is alingned with shifted_point2 on the gap direction
    mid = shifted_point2 + (gap_offset * direction).astype(int)
    mid = mid + (gap_offset * direction / 2).astype(int)
    # waypoints = [shifted_point1, shifted_point2, center_point, outer_waypoint1, outer_waypoint2, outer_waypoint3, mid]
    # save the waypoints in a csv file in x, y. ignore names
    with open('waypoints.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['x', 'y', 'yaw'])
        writer.writerow([shifted_point1[0], shifted_point1[1], angle])
        writer.writerow([center_point[0], center_point[1], angle])
        writer.writerow([shifted_point2[0], shifted_point2[1], angle])
        writer.writerow([mid[0], mid[1], angle+angle_offset])
        writer.writerow([outer_waypoint1[0], outer_waypoint1[1], angle])
        writer.writerow([outer_waypoint2[0], outer_waypoint2[1], angle])
        writer.writerow([outer_waypoint3[0], outer_waypoint3[1], angle])
        
    return shifted_point1, shifted_point2, center_point, outer_waypoint1, outer_waypoint2, outer_waypoint3, mid



# Example usage
if __name__ == "__main__":
    # Load the binary map
    binary_map = cv2.imread("/home/yufeiyang/Documents/paralle-parking-roboracer/maps/agh/agh300a1_blacked.pgm", cv2.IMREAD_GRAYSCALE)
    
    # Set the minimum gap width threshold
    GAP_THRESHOLD = 20  # Adjust based on your map scale
    UPPER_THRESHOLD = 40  # Maximum gap width to consider
    # Detect and visualize gaps
    gaps = visualize_gaps(binary_map, GAP_THRESHOLD, UPPER_THRESHOLD)

