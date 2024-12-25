import cv2
import numpy as np
from math import exp, pi

from utils import *

def main():

    # Video capture
    cap = cv2.VideoCapture("data/video_1.mp4")
    if not cap.isOpened():
        print("Error: Cannot open video.")
        exit()

    # Define parameters
    min_contour_area = 500
    min_contour_width = 25
    min_contour_height = 25
    vehicle_count = 0
    prev_count = 0
    vehicle_types = {
        'Car[s]': 0,
        'Transit[m]': 0,
        'Truck[b]': 0,
    }

    # Road area parameters
    x_start, x_end = 10, 440
    y_start, y_end = 240, 440
    line_position = (y_start + y_end) // 2

    # Tracking parameters
    active_vehicles = []  # List to store currently tracked vehicles
    tracking_memory = 10  # Number of frames to keep tracking a vehicle
    min_tracking_distance = 50  # Minimum distance to consider it's the same vehicle

    # Read the first frame for initialization
    ret, first_frame = cap.read()
    if not ret:
        print("Error: Cannot read video.")
        cap.release()
        exit()

    # Crop and create ROI mask
    first_frame = first_frame[0:450, 0:450]

    roi_mask = create_roi_mask(first_frame, x_start, x_end, y_start, y_end)

    # Background initialization
    first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    first_gray = gaussian_blur(first_gray, 5, sigma=0)
    first_gray = bitwise_and(first_gray, first_gray, mask=roi_mask)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video.")
            break

        # Crop frame
        frame = frame[0:450, 0:450]

        # Process frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = gaussian_blur(gray, 5, sigma=0)
        gray = bitwise_and(gray, gray, mask=roi_mask)

        # Compute difference and threshold
        frame_diff = absdiff(first_gray, gray)

        if not use_otsu_thres:
            ret, thresh = threshold(frame_diff, 30, 255)
        else:
            thresh, optimal_threshold = otsu_thresholding(frame_diff)
            # print(optimal_threshold)

        dilated = dilate(thresh, create_rectangular_kernel(3), iterations=4)
        eroded = erode(dilated, create_rectangular_kernel(3), iterations=8)

        image_contours = eroded.copy()

        if not use_contour_stratch:
            contours, _ = cv2.findContours(image_contours, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        else:
            contours = find_contours_from_scratch(image_contours.copy())

        # Draw ROI area
        cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
        cv2.line(frame, (x_start, line_position), (x_end, line_position), (255, 0, 0), 2)

        # List to store current frame's vehicles
        current_vehicles = []

        for contour in contours:
            # x, y, w, h = cv2.boundingRect(contour)
            x, y, w, h = bounding_rect_nested(contour)

            if (x >= x_start and x + w <= x_end and y >= y_start and y + h <= y_end):

                aspect_ratio = float(w) / h

                # area_ = cv2.contourArea(contour)
                area_ = contour_area_nested(contour)

                if (area_ >= min_contour_area and
                        w >= min_contour_width and
                        h >= min_contour_height and
                        0.4 <= aspect_ratio <= 2.5):
                    # Calculate center point of vehicle
                    center_x = x + w // 2
                    center_y = y + h // 2

                    # Store current vehicle
                    current_vehicles.append({
                        'center': (center_x, center_y),
                        'bbox': (x, y, w, h),
                        'frames_tracked': 0,
                        'counted': False
                    })

        # Update tracking for existing vehicles
        new_active_vehicles = []

        for active in active_vehicles:
            active_center = active['center']
            matched = False

            # Try to match with current vehicles
            for current in current_vehicles:
                current_center = current['center']

                # Calculate distance between centers
                distance = np.sqrt((active_center[0] - current_center[0]) ** 2 +
                                   (active_center[1] - current_center[1]) ** 2)

                if distance < min_tracking_distance:
                    # Update existing vehicle
                    active.update({
                        'center': current_center,
                        'bbox': current['bbox'],
                        'frames_tracked': active['frames_tracked'] + 1
                    })
                    current_vehicles.remove(current)
                    matched = True
                    break

            # Keep vehicle if it's still being tracked
            if matched and active['frames_tracked'] < tracking_memory:
                new_active_vehicles.append(active)

                # Check if vehicle crosses the counting line
                if (not active['counted'] and
                        line_position - 5 < active['center'][1] < line_position + 5):
                    vehicle_count += 1
                    active['counted'] = True

                    area = active['bbox'][2] * active['bbox'][3]

                    if area < 6000:
                        vehicle_types['Car[s]'] += 1
                    elif area > 6000 and area < 20000:
                        vehicle_types['Transit[m]'] += 1
                    else:
                        vehicle_types['Truck[b]'] += 1

        # Add new vehicles to tracking
        for vehicle in current_vehicles:
            new_active_vehicles.append(vehicle)

        active_vehicles = new_active_vehicles

        # Draw bounding boxes for tracked vehicles
        for vehicle in active_vehicles:
            x, y, w, h = vehicle['bbox']
            area = w * h

            # color = (0, 255, 0) if vehicle['counted'] else (0, 0, 255)
            if vehicle['counted']:
                color = (0, 255, 0)
                cv2.putText(frame, f"Area: [{area}]", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                if prev_count != vehicle_count:
                    print(prev_count, f"Area: [{area}]")

            else:
                color = (0, 0, 255)
                # cv2.putText(frame, f"Area: [{w * h}]", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # Display information
        cv2.putText(frame, f"Vehicle Count: {vehicle_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, f"Car[s]: {vehicle_types['Car[s]']}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.putText(frame, f"Transit[m]: {vehicle_types['Transit[m]']}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.putText(frame, f"Truck[b]: {vehicle_types['Truck[b]']}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        # Display frames
        cv2.imshow('before contours', image_contours)
        cv2.imshow('Vehicle Detection', frame)

        prev_count = vehicle_count

        if cv2.waitKey(30) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":

    use_otsu_thres = False
    use_contour_stratch = False
    main()