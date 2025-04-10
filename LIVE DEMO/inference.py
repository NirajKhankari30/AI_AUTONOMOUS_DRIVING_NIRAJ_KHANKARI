import cv2
import time
import os
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

# Add new imports for tracking
from scipy.spatial import distance

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

def process_lane_detection(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply Canny edge detection
    edges = cv2.Canny(blur, 50, 150)
    
    # Create a mask for the region of interest (ROI)
    height, width = frame.shape[:2]
    mask = np.zeros_like(edges)
    
    # Define a more precise trapezoid ROI
    roi_vertices = np.array([[
        (width * 0.1, height),           # Bottom left
        (width * 0.45, height * 0.6),    # Top left
        (width * 0.55, height * 0.6),    # Top right
        (width * 0.9, height)            # Bottom right
    ]], dtype=np.int32)
    
    cv2.fillPoly(mask, roi_vertices, 255)
    masked_edges = cv2.bitwise_and(edges, mask)
    
    # Apply Hough Transform for line detection with adjusted parameters
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, 50, 
                           minLineLength=50, maxLineGap=100)
    
    return lines, height, width, roi_vertices

def draw_lanes(frame, lines, height, width, roi_vertices):
    if lines is None:
        return frame
    
    # Create a blank image for lanes
    lane_image = np.zeros_like(frame)
    
    # Draw ROI area with semi-transparent overlay
    roi_overlay = frame.copy()
    cv2.fillPoly(roi_overlay, roi_vertices, (0, 255, 0))
    frame = cv2.addWeighted(frame, 0.9, roi_overlay, 0.1, 0)
    
    # Separate left and right lines
    left_lines = []
    right_lines = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 - x1 == 0:
            continue
            
        slope = (y2 - y1) / (x2 - x1)
        
        # Filter lines based on slope
        if slope < -0.3:  # Left lane
            left_lines.append(line[0])
        elif slope > 0.3:  # Right lane
            right_lines.append(line[0])
    
    # Draw left lane
    if left_lines:
        left_line = np.mean(left_lines, axis=0, dtype=np.int32)
        # Draw filled polygon for left lane
        pts_left = np.array([[left_line[0], left_line[1]], 
                           [left_line[2], left_line[3]], 
                           [left_line[2], height],
                           [left_line[0], height]], np.int32)
        cv2.fillPoly(lane_image, [pts_left], (0, 0, 255))
    
    # Draw right lane
    if right_lines:
        right_line = np.mean(right_lines, axis=0, dtype=np.int32)
        # Draw filled polygon for right lane
        pts_right = np.array([[right_line[0], right_line[1]], 
                            [right_line[2], right_line[3]], 
                            [right_line[2], height],
                            [right_line[0], height]], np.int32)
        cv2.fillPoly(lane_image, [pts_right], (255, 0, 0))
    
    # Add lane overlay with transparency
    result = cv2.addWeighted(frame, 1, lane_image, 0.3, 0)
    
    # Draw lane boundaries with solid lines
    if left_lines:
        cv2.line(result, (left_line[0], left_line[1]), 
                (left_line[2], left_line[3]), (0, 0, 255), 3)
    if right_lines:
        cv2.line(result, (right_line[0], right_line[1]), 
                (right_line[2], right_line[3]), (255, 0, 0), 3)
    
    return result

def calculate_steering_angle(lines, width):
    if lines is None or len(lines) < 2:
        return 0, "CENTER"
    
    left_lines = []
    right_lines = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 - x1 == 0:
            continue
        slope = (y2 - y1) / (x2 - x1)
        if slope < -0.3:
            left_lines.append(line[0])
        elif slope > 0.3:
            right_lines.append(line[0])
    
    if not left_lines or not right_lines:
        return 0, "CENTER"
    
    # Calculate lane center
    left_center = np.mean(left_lines, axis=0)
    right_center = np.mean(right_lines, axis=0)
    lane_center = (left_center[0] + right_center[0]) / 2
    
    # Calculate steering angle
    image_center = width / 2
    offset = lane_center - image_center
    steering_angle = offset / (width / 2) * 45
    
    # Determine direction
    if abs(steering_angle) < 5:
        direction = "CENTER"
    elif steering_angle < 0:
        direction = "LEFT"
    else:
        direction = "RIGHT"
    
    return steering_angle, direction

def calculate_trapezoid_area(vertices):
    """
    Calculate the area of a trapezoid given its vertices.
    Args:
        vertices: numpy array of shape (1, 4, 2) containing the vertices of the trapezoid
    Returns:
        float: Area of the trapezoid
    """
    # Extract the four vertices
    bottom_left = vertices[0][0]
    top_left = vertices[0][1]
    top_right = vertices[0][2]
    bottom_right = vertices[0][3]
    
    # Calculate the lengths of the parallel sides (top and bottom)
    top_width = abs(top_right[0] - top_left[0])
    bottom_width = abs(bottom_right[0] - bottom_left[0])
    
    # Calculate the height (average of left and right heights)
    left_height = abs(bottom_left[1] - top_left[1])
    right_height = abs(bottom_right[1] - top_right[1])
    height = (left_height + right_height) / 2
    
    # Calculate area using trapezoid formula
    area = ((top_width + bottom_width) / 2) * height
    return area

def analyze_traffic_density(total_vehicles, roi_area):
    """
    Analyze traffic density based on the number of vehicles and ROI area.
    Args:
        total_vehicles: int, total number of vehicles detected
        roi_area: float, area of the region of interest
    Returns:
        tuple: (density, density_level)
    """
    if total_vehicles == 0:
        return 0, "Low"
    
    density = total_vehicles / roi_area
    
    if density < 0.001:
        density_level = "Low"
    elif density < 0.005:
        density_level = "Medium"
    else:
        density_level = "High"
    
    return density, density_level

class VehicleTracker:
    def __init__(self, max_disappeared=30):
        self.next_object_id = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
        self.speeds = defaultdict(list)
        self.positions = defaultdict(list)
        self.timestamps = defaultdict(list)

    def update(self, detections, timestamp):
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    del self.objects[object_id]
                    del self.disappeared[object_id]
            return self.objects

        if len(self.objects) == 0:
            for detection in detections:
                self.register(detection, timestamp)
        else:
            self.update_existing(detections, timestamp)

        return self.objects

    def register(self, detection, timestamp):
        self.objects[self.next_object_id] = detection
        self.disappeared[self.next_object_id] = 0
        self.positions[self.next_object_id].append(detection)
        self.timestamps[self.next_object_id].append(timestamp)
        self.next_object_id += 1

    def update_existing(self, detections, timestamp):
        object_ids = list(self.objects.keys())
        object_centers = [self.get_center(self.objects[object_id]) for object_id in object_ids]
        detection_centers = [self.get_center(detection) for detection in detections]

        D = distance.cdist(np.array(object_centers), np.array(detection_centers))
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows = set()
        used_cols = set()

        for (row, col) in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue

            object_id = object_ids[row]
            self.objects[object_id] = detections[col]
            self.disappeared[object_id] = 0
            self.positions[object_id].append(detections[col])
            self.timestamps[object_id].append(timestamp)

            used_rows.add(row)
            used_cols.add(col)

        unused_rows = set(range(0, D.shape[0])).difference(used_rows)
        unused_cols = set(range(0, D.shape[1])).difference(used_cols)

        for row in unused_rows:
            object_id = object_ids[row]
            self.disappeared[object_id] += 1
            if self.disappeared[object_id] > self.max_disappeared:
                del self.objects[object_id]
                del self.disappeared[object_id]

        for col in unused_cols:
            self.register(detections[col], timestamp)

    def get_center(self, detection):
        x1, y1, x2, y2 = detection
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def calculate_speed(self, object_id, pixels_per_meter=50):
        if len(self.positions[object_id]) < 2:
            return 0

        positions = self.positions[object_id]
        timestamps = self.timestamps[object_id]
        
        # Calculate distance in pixels
        current_pos = self.get_center(positions[-1])
        prev_pos = self.get_center(positions[-2])
        pixel_distance = distance.euclidean(current_pos, prev_pos)
        
        # Calculate time difference
        time_diff = timestamps[-1] - timestamps[-2]
        if time_diff == 0:
            return 0
        
        # Convert to meters per second
        speed_mps = (pixel_distance / pixels_per_meter) / time_diff
        # Convert to km/h
        speed_kmh = speed_mps * 3.6
        
        return speed_kmh

def calculate_collision_warning(vehicle_tracker, frame_shape, warning_distance=50):
    warnings = []
    height, width = frame_shape[:2]
    
    for object_id, detection in vehicle_tracker.objects.items():
        x1, y1, x2, y2 = detection
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # Calculate distance from center of frame
        frame_center_x = width / 2
        frame_center_y = height / 2
        dist = distance.euclidean((center_x, center_y), (frame_center_x, frame_center_y))
        
        if dist < warning_distance:
            speed = vehicle_tracker.calculate_speed(object_id)
            warnings.append({
                'object_id': object_id,
                'distance': dist,
                'speed': speed,
                'position': (int(center_x), int(center_y))
            })
    
    return warnings

def draw_dashboard(frame, fps, steering_angle, direction, detections, safe_distance=True,
                  car_count=0, truck_count=0, bus_count=0, person_count=0, density_level="Low",
                  collision_warnings=None, vehicle_speeds=None):
    height, width = frame.shape[:2]
    dashboard_height = 150
    dashboard = np.zeros((dashboard_height, width, 3), dtype=np.uint8)
    
    # Draw dashboard background
    cv2.rectangle(dashboard, (0, 0), (width, dashboard_height), (50, 50, 50), -1)
    
    # Draw FPS meter
    fps_text = f"FPS: {fps:.1f}"
    cv2.putText(dashboard, fps_text, (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Draw steering visualization
    center_x = width // 2
    wheel_y = dashboard_height // 2
    
    # Draw steering wheel
    cv2.circle(dashboard, (center_x, wheel_y), 40, (255, 255, 255), 2)
    
    # Draw steering direction indicator
    angle_rad = np.radians(steering_angle)
    end_x = int(center_x + 40 * np.sin(angle_rad))
    end_y = int(wheel_y - 40 * np.cos(angle_rad))
    cv2.line(dashboard, (center_x, wheel_y), (end_x, end_y), (0, 255, 0), 3)
    
    # Draw direction text with color coding
    direction_colors = {
        "LEFT": (0, 0, 255),
        "CENTER": (0, 255, 0),
        "RIGHT": (255, 0, 0)
    }
    direction_text = f"Steering: {direction} ({steering_angle:.1f}°)"
    cv2.putText(dashboard, direction_text, (center_x + 100, wheel_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, direction_colors[direction], 2)
    
    # Draw detection count
    det_text = f"Detections: {detections}"
    cv2.putText(dashboard, det_text, (width - 250, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    
    # Draw traffic insights
    traffic_x = width - 400
    traffic_y = 80
    line_height = 30
    
    # Object counts
    cv2.putText(dashboard, f"Cars: {car_count}", (traffic_x, traffic_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(dashboard, f"Trucks: {truck_count}", (traffic_x, traffic_y + line_height), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(dashboard, f"Buses: {bus_count}", (traffic_x, traffic_y + 2*line_height), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(dashboard, f"Persons: {person_count}", (traffic_x, traffic_y + 3*line_height), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Traffic density
    density_color = (0, 255, 0) if density_level == "Low" else (0, 255, 255) if density_level == "Medium" else (0, 0, 255)
    cv2.putText(dashboard, f"Traffic Density: {density_level}", (traffic_x, traffic_y + 4*line_height), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, density_color, 2)
    
    # Draw collision warnings
    if collision_warnings:
        warning_x = 20
        warning_y = traffic_y
        for warning in collision_warnings:
            warning_text = f"Warning! Object {warning['object_id']}: {warning['distance']:.1f}m"
            cv2.putText(dashboard, warning_text, (warning_x, warning_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            warning_y += line_height
    
    # Draw vehicle speeds
    if vehicle_speeds:
        speed_x = width - 600
        speed_y = traffic_y
        for obj_id, speed in vehicle_speeds.items():
            speed_text = f"Vehicle {obj_id}: {speed:.1f} km/h"
            cv2.putText(dashboard, speed_text, (speed_x, speed_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            speed_y += line_height
    
    # Combine dashboard with frame
    result = np.vstack((frame, dashboard))
    return result

def create_pdf_report(detection_stats, lane_stats, speed_stats, traffic_stats, processing_stats, self_driving_stats, output_path):
    """
    Create a PDF report with all processing information.
    Args:
        detection_stats: Dictionary containing detection statistics
        lane_stats: Dictionary containing lane detection statistics
        speed_stats: Dictionary containing speed analysis statistics
        traffic_stats: Dictionary containing traffic analysis statistics
        processing_stats: Dictionary containing processing performance statistics
        self_driving_stats: Dictionary containing self-driving performance statistics
        output_path: Path where to save the PDF
    """
    # Create the PDF document
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Add title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=1  # Center alignment
    )
    story.append(Paragraph("Video Processing Summary Report", title_style))
    
    # 1. Object Detection Statistics
    story.append(Paragraph("1. Object Detection Statistics", styles['Heading2']))
    story.append(Spacer(1, 12))
    
    # Create detection statistics table
    detection_data = [['Class', 'Count', 'Confidence Distribution']]
    for class_name, stats in detection_stats.items():
        conf_dist = ', '.join([f"{range}: {count}" for range, count in stats['confidence_distribution'].items()])
        detection_data.append([class_name, str(stats['count']), conf_dist])
    
    detection_table = Table(detection_data, colWidths=[2*inch, 1*inch, 3*inch])
    detection_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(detection_table)
    story.append(Spacer(1, 20))
    
    # 2. Lane Detection Statistics
    story.append(Paragraph("2. Lane Detection Statistics", styles['Heading2']))
    story.append(Spacer(1, 12))
    
    lane_data = [['Metric', 'Value']]
    for stat_name, value in lane_stats.items():
        lane_data.append([stat_name, str(value)])
    
    lane_table = Table(lane_data, colWidths=[3*inch, 2*inch])
    lane_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(lane_table)
    story.append(Spacer(1, 20))
    
    # 3. Speed Analysis
    story.append(Paragraph("3. Speed Analysis", styles['Heading2']))
    story.append(Spacer(1, 12))
    
    speed_data = [['Metric', 'Value']]
    for stat_name, value in speed_stats.items():
        speed_data.append([stat_name, str(value)])
    
    speed_table = Table(speed_data, colWidths=[3*inch, 2*inch])
    speed_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(speed_table)
    story.append(Spacer(1, 20))
    
    # 4. Traffic Analysis
    story.append(Paragraph("4. Traffic Analysis", styles['Heading2']))
    story.append(Spacer(1, 12))
    
    traffic_data = [['Metric', 'Value']]
    for stat_name, value in traffic_stats.items():
        traffic_data.append([stat_name, str(value)])
    
    traffic_table = Table(traffic_data, colWidths=[3*inch, 2*inch])
    traffic_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(traffic_table)
    story.append(Spacer(1, 20))
    
    # 5. Processing Performance
    story.append(Paragraph("5. Processing Performance", styles['Heading2']))
    story.append(Spacer(1, 12))
    
    processing_data = [['Metric', 'Value']]
    for stat_name, value in processing_stats.items():
        processing_data.append([stat_name, str(value)])
    
    processing_table = Table(processing_data, colWidths=[3*inch, 2*inch])
    processing_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(processing_table)
    story.append(Spacer(1, 20))
    
    # 6. Self-Driving Performance Analysis
    story.append(Paragraph("6. Self-Driving Performance Analysis", styles['Heading2']))
    story.append(Spacer(1, 12))
    
    # Lane Keeping Performance
    story.append(Paragraph("Lane Keeping Performance", styles['Heading3']))
    story.append(Spacer(1, 12))
    
    lane_performance_data = [['Metric', 'Value']]
    lane_performance = self_driving_stats['lane_keeping']
    lane_performance_data.extend([
        ['Lane Centering Accuracy', f"{lane_performance['centering_accuracy']:.1f}%"],
        ['Lane Departure Events', str(lane_performance['departure_count'])],
        ['Average Lane Position Error', f"{lane_performance['avg_position_error']:.1f} pixels"]
    ])
    
    lane_performance_table = Table(lane_performance_data, colWidths=[3*inch, 2*inch])
    lane_performance_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(lane_performance_table)
    story.append(Spacer(1, 20))
    
    # Steering Behavior
    story.append(Paragraph("Steering Behavior", styles['Heading3']))
    story.append(Spacer(1, 12))
    
    steering_data = [['Metric', 'Value']]
    steering = self_driving_stats['steering']
    steering_data.extend([
        ['Average Steering Angle', f"{steering['avg_angle']:.1f}°"],
        ['Steering Smoothness', f"{steering['smoothness']:.1f}%"],
        ['Sharp Turns', str(steering['sharp_turns'])]
    ])
    
    steering_table = Table(steering_data, colWidths=[3*inch, 2*inch])
    steering_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(steering_table)
    story.append(Spacer(1, 20))
    
    # Safety Metrics
    story.append(Paragraph("Safety Metrics", styles['Heading3']))
    story.append(Spacer(1, 12))
    
    safety_data = [['Metric', 'Value']]
    safety = self_driving_stats['safety']
    safety_data.extend([
        ['Collision Warnings', str(safety['collision_warnings'])],
        ['Safe Distance Violations', str(safety['distance_violations'])],
        ['Emergency Maneuvers', str(safety['emergency_maneuvers'])]
    ])
    
    safety_table = Table(safety_data, colWidths=[3*inch, 2*inch])
    safety_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(safety_table)
    story.append(Spacer(1, 20))
    
    # System Reliability
    story.append(Paragraph("System Reliability", styles['Heading3']))
    story.append(Spacer(1, 12))
    
    reliability_data = [['Metric', 'Value']]
    reliability = self_driving_stats['reliability']
    reliability_data.extend([
        ['Lane Detection Success Rate', f"{reliability['lane_detection_rate']:.1f}%"],
        ['Object Detection Success Rate', f"{reliability['object_detection_rate']:.1f}%"],
        ['System Uptime', f"{reliability['uptime']:.1f}%"]
    ])
    
    reliability_table = Table(reliability_data, colWidths=[3*inch, 2*inch])
    reliability_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(reliability_table)
    
    # Build the PDF
    doc.build(story)
    print(f"PDF report saved to: {output_path}")

try:
    # Initialize video capture and model
    print(f"OpenCV version: {cv2.__version__}")
    print(f"CUDA is {'not ' if not cv2.cuda.getCudaEnabledDeviceCount() else ''}available")
    
    print("Loading YOLO model...")
    model = YOLO("m_model.pt")
    print("Model loaded successfully!")
    print(f"Model classes: {model.names}")

    # Initialize vehicle tracker
    vehicle_tracker = VehicleTracker()

    # Initialize statistics tracking
    detection_stats = {class_name: {'count': 0, 'confidence_distribution': {
        '0.0-0.2': 0, '0.2-0.4': 0, '0.4-0.6': 0, '0.6-0.8': 0, '0.8-1.0': 0
    }} for class_name in model.names.values()}
    
    lane_stats = {
        'Total Lane Detections': 0,
        'Average Lane Width': 0,
        'Lane Departure Count': 0,
        'Average Steering Angle': 0
    }
    
    speed_stats = {
        'Average Vehicle Speed': 0,
        'Maximum Speed Detected': 0,
        'Speed Violation Count': 0
    }
    
    traffic_stats = {
        'Average Traffic Density': 0,
        'Maximum Traffic Density': 0,
        'Traffic Jam Duration': 0
    }
    
    processing_stats = {
        'Total Frames Processed': 0,
        'Average FPS': 0,
        'Processing Time': 0
    }
    
    # Initialize self-driving performance statistics
    self_driving_stats = {
        'lane_keeping': {
            'centering_accuracy': 0,
            'departure_count': 0,
            'avg_position_error': 0
        },
        'steering': {
            'avg_angle': 0,
            'smoothness': 0,
            'sharp_turns': 0
        },
        'safety': {
            'collision_warnings': 0,
            'distance_violations': 0,
            'emergency_maneuvers': 0
        },
        'reliability': {
            'lane_detection_rate': 0,
            'object_detection_rate': 0,
            'uptime': 0
        }
    }

    video_path = "test_1.mp4"
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file {video_path} not found")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("Error: Could not open video file")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties:")
    print(f" - Resolution: {width}x{height}")
    print(f" - FPS: {fps}")
    print(f" - Total frames: {total_frames}")

    # Create window
    window_name = "Advanced Autonomous Driving System"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 900)  # Adjusted for dashboard

    # Initialize colors for object detection
    COLORS = np.random.uniform(0, 255, size=(100, 3))

    # Initialize traffic analysis counters
    car_count = 0
    truck_count = 0
    bus_count = 0
    person_count = 0
    total_vehicles = 0

    frame_count = 0
    start_time = time.time()
    last_print_time = start_time

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"End of video reached after {frame_count} frames")
            break

        try:
            # Process lane detection
            lines, height, width, roi = process_lane_detection(frame)
            
            # Calculate steering
            steering_angle, direction = calculate_steering_angle(lines, width)
            
            # Draw lanes
            frame_with_lanes = draw_lanes(frame, lines, height, width, roi)
            
            # Run YOLO inference
            results = model(frame, conf=0.25)
            
            # Reset counters for new frame
            car_count = 0
            truck_count = 0
            bus_count = 0
            person_count = 0
            
            # Process detections
            num_detections = 0
            detections = []
            
            for r in results:
                boxes = r.boxes
                num_detections = len(boxes)
                
                for box in boxes:
                    # Get detection info
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    class_name = model.names[cls]
                    
                    # Store detection for tracking
                    detections.append((x1, y1, x2, y2))
                    
                    # Update object counters
                    if class_name == 'car':
                        car_count += 1
                    elif class_name == 'truck':
                        truck_count += 1
                    elif class_name == 'bus':
                        bus_count += 1
                    elif class_name == 'person':
                        person_count += 1
                    
                    # Generate color
                    color = tuple(map(int, COLORS[cls % len(COLORS)]))
                    
                    # Draw box and label
                    cv2.rectangle(frame_with_lanes, (x1, y1), (x2, y2), color, 2)
                    label = f'{class_name} {conf:.2f}'
                    
                    # Add label with background
                    (label_width, label_height), _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    cv2.rectangle(
                        frame_with_lanes,
                        (x1, y1 - label_height - 5),
                        (x1 + label_width, y1),
                        color,
                        -1
                    )
                    cv2.putText(
                        frame_with_lanes,
                        label,
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        2
                    )
            
            # Update vehicle tracker
            current_time = time.time()
            vehicle_tracker.update(detections, current_time)
            
            # Calculate vehicle speeds
            vehicle_speeds = {}
            for obj_id in vehicle_tracker.objects.keys():
                speed = vehicle_tracker.calculate_speed(obj_id)
                if speed > 0:  # Only show moving vehicles
                    vehicle_speeds[obj_id] = speed
            
            # Calculate collision warnings
            collision_warnings = calculate_collision_warning(vehicle_tracker, frame.shape)
            
            # Calculate total vehicles
            total_vehicles = car_count + truck_count + bus_count
            
            # Calculate traffic density
            roi_area = calculate_trapezoid_area(roi)
            _, density_level = analyze_traffic_density(total_vehicles, roi_area)
            
            # Calculate current FPS
            current_time = time.time()
            elapsed_time = current_time - start_time
            current_fps = frame_count / elapsed_time
            
            # Add dashboard with traffic insights
            final_frame = draw_dashboard(
                frame_with_lanes, 
                current_fps, 
                steering_angle, 
                direction, 
                num_detections,
                True,
                car_count,
                truck_count,
                bus_count,
                person_count,
                density_level,
                collision_warnings,
                vehicle_speeds
            )
            
            # Display the result
            cv2.imshow(window_name, final_frame)
            
            frame_count += 1
            
            # Print stats every second
            if current_time - last_print_time >= 1.0:
                print(f"Frame {frame_count}/{total_frames} - "
                      f"FPS: {current_fps:.2f} - "
                      f"Detections: {num_detections} - "
                      f"Steering: {direction} ({steering_angle:.1f}°) - "
                      f"Traffic Density: {density_level}")
                last_print_time = current_time

            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Quitting...")
                break
            
            # Update statistics
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    class_name = model.names[cls]
                    
                    # Update detection statistics
                    detection_stats[class_name]['count'] += 1
                    conf_range = f"{int(conf*5)/5:.1f}-{int(conf*5)/5+0.2:.1f}"
                    detection_stats[class_name]['confidence_distribution'][conf_range] += 1
            
            # Update lane statistics
            if lines is not None:
                lane_stats['Total Lane Detections'] += 1
                lane_stats['Average Steering Angle'] = (lane_stats['Average Steering Angle'] * 
                                                      (lane_stats['Total Lane Detections'] - 1) + 
                                                      abs(steering_angle)) / lane_stats['Total Lane Detections']
            
            # Update speed statistics
            for obj_id, speed in vehicle_speeds.items():
                speed_stats['Average Vehicle Speed'] = (speed_stats['Average Vehicle Speed'] * 
                                                      len(vehicle_speeds) + speed) / (len(vehicle_speeds) + 1)
                speed_stats['Maximum Speed Detected'] = max(speed_stats['Maximum Speed Detected'], speed)
                if speed > 80:  # Assuming 80 km/h as speed limit
                    speed_stats['Speed Violation Count'] += 1
            
            # Update traffic statistics
            _, density_level = analyze_traffic_density(total_vehicles, roi_area)
            traffic_stats['Average Traffic Density'] = (traffic_stats['Average Traffic Density'] * 
                                                      frame_count + total_vehicles) / (frame_count + 1)
            traffic_stats['Maximum Traffic Density'] = max(traffic_stats['Maximum Traffic Density'], total_vehicles)
            if density_level == "High":
                traffic_stats['Traffic Jam Duration'] += 1
            
            # Update processing statistics
            processing_stats['Total Frames Processed'] = frame_count
            processing_stats['Average FPS'] = current_fps
            processing_stats['Processing Time'] = elapsed_time

            # Update self-driving performance statistics
            # Lane keeping performance
            if lines is not None:
                lane_center = width / 2
                position_error = abs(steering_angle) / 45 * 100  # Convert angle to position error
                self_driving_stats['lane_keeping']['avg_position_error'] = (
                    self_driving_stats['lane_keeping']['avg_position_error'] * 
                    self_driving_stats['lane_keeping']['departure_count'] + 
                    position_error
                ) / (self_driving_stats['lane_keeping']['departure_count'] + 1)
                
                if abs(steering_angle) > 15:  # Threshold for lane departure
                    self_driving_stats['lane_keeping']['departure_count'] += 1
            
            # Steering behavior
            self_driving_stats['steering']['avg_angle'] = (
                self_driving_stats['steering']['avg_angle'] * frame_count + 
                abs(steering_angle)
            ) / (frame_count + 1)
            
            if abs(steering_angle) > 30:  # Threshold for sharp turns
                self_driving_stats['steering']['sharp_turns'] += 1
            
            # Safety metrics
            if collision_warnings:
                self_driving_stats['safety']['collision_warnings'] += len(collision_warnings)
            
            for warning in collision_warnings:
                if warning['distance'] < 30:  # Threshold for distance violation
                    self_driving_stats['safety']['distance_violations'] += 1
                if warning['speed'] > 80:  # Threshold for emergency maneuver
                    self_driving_stats['safety']['emergency_maneuvers'] += 1
            
            # System reliability
            self_driving_stats['reliability']['lane_detection_rate'] = (
                self_driving_stats['reliability']['lane_detection_rate'] * frame_count + 
                (100 if lines is not None else 0)
            ) / (frame_count + 1)
            
            self_driving_stats['reliability']['object_detection_rate'] = (
                self_driving_stats['reliability']['object_detection_rate'] * frame_count + 
                (100 if num_detections > 0 else 0)
            ) / (frame_count + 1)
            
            self_driving_stats['reliability']['uptime'] = 100  # Assuming continuous operation

        except Exception as e:
            print(f"Error processing frame {frame_count}: {str(e)}")
            continue

    # After video processing is complete
    # Create and save PDF report
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = f"processing_summary_{timestamp}.pdf"
    create_pdf_report(
        detection_stats,
        lane_stats,
        speed_stats,
        traffic_stats,
        processing_stats,
        self_driving_stats,
        output_path
    )
    
    print("\nProcessing completed. Summary report has been saved as a PDF file.")
    print(f"Please check the file: {output_path}")

except Exception as e:
    print(f"An error occurred: {str(e)}")

finally:
    if 'cap' in locals():
        cap.release()
    cv2.destroyAllWindows()
    
    if 'frame_count' in locals() and 'start_time' in locals():
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time
        print(f"\nProcessing completed:")
        print(f" - Total frames processed: {frame_count}")
        print(f" - Average FPS: {avg_fps:.2f}")
        print(f" - Total time: {total_time:.2f} seconds")