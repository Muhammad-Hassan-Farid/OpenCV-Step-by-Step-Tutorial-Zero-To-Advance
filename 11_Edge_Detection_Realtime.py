"""
Task 4: Apply Edge Detection to a Real-World Scene
Scenario: Capture an image using a webcam and detect edges in real-time.

This script demonstrates real-time edge detection using OpenCV's Canny edge detector.
The program captures video from your webcam and applies edge detection to each frame.
"""

import cv2
import numpy as np


def main():
    """
    Main function to capture webcam feed and apply real-time edge detection.
    """
    # Initialize the webcam (0 is the default camera)
    print("Initializing webcam...")
    cap = cv2.VideoCapture(0)
    
    # Check if webcam is opened successfully
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    print("Webcam initialized successfully!")
    print("Press 'q' to quit")
    print("Press 's' to save current frame and edges")
    print("Press '+' to increase edge sensitivity (decrease threshold)")
    print("Press '-' to decrease edge sensitivity (increase threshold)")
    
    # Initial Canny edge detection thresholds
    threshold1 = 50
    threshold2 = 150
    
    # Counter for saved images
    save_counter = 0
    
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        # If frame is read correctly, ret is True
        if not ret:
            print("Error: Can't receive frame. Exiting...")
            break
        
        # Convert the frame to grayscale (Canny edge detection works on grayscale images)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian Blur to reduce noise and improve edge detection
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
        
        # Apply Canny Edge Detection
        edges = cv2.Canny(blurred, threshold1, threshold2)
        
        # Create a colored version of edges (for better visualization)
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        # Optional: Apply edges as an overlay on the original frame
        # Create a mask where edges are highlighted in green
        overlay = frame.copy()
        overlay[edges > 0] = [0, 255, 0]  # Green color for edges
        blended = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Add text information to the frames
        cv2.putText(frame, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 255, 0), 2)
        cv2.putText(edges_colored, "Edges Detected", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(edges_colored, f"Threshold: {threshold1}/{threshold2}", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(blended, "Overlay View", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 255, 0), 2)
        
        # Stack images horizontally for comparison
        top_row = np.hstack([frame, edges_colored])
        bottom_row = np.hstack([blended, cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)])
        combined = np.vstack([top_row, bottom_row])
        
        # Display the resulting frames
        cv2.imshow('Real-Time Edge Detection', combined)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            # Quit the program
            print("Exiting...")
            break
        elif key == ord('s'):
            # Save the current frame and edges
            save_counter += 1
            cv2.imwrite(f'original_frame_{save_counter}.jpg', frame)
            cv2.imwrite(f'edges_frame_{save_counter}.jpg', edges)
            cv2.imwrite(f'overlay_frame_{save_counter}.jpg', blended)
            print(f"Saved frame {save_counter}")
        elif key == ord('+') or key == ord('='):
            # Increase sensitivity (decrease thresholds)
            threshold1 = max(10, threshold1 - 10)
            threshold2 = max(20, threshold2 - 10)
            print(f"Increased sensitivity: {threshold1}/{threshold2}")
        elif key == ord('-') or key == ord('_'):
            # Decrease sensitivity (increase thresholds)
            threshold1 = min(200, threshold1 + 10)
            threshold2 = min(300, threshold2 + 10)
            print(f"Decreased sensitivity: {threshold1}/{threshold2}")
    
    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam released and windows closed.")


def demonstrate_edge_detection_techniques():
    """
    Bonus function to demonstrate different edge detection techniques on a single image.
    """
    print("\n--- Edge Detection Techniques Demo ---")
    print("This function can be called separately to test on a saved image.")
    
    # Capture a single frame
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Could not capture image from webcam")
        return
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
    
    # 1. Canny Edge Detection (most popular)
    canny = cv2.Canny(blurred, 50, 150)
    
    # 2. Sobel Edge Detection (X and Y directions)
    sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    sobel = np.sqrt(sobelx**2 + sobely**2)
    sobel = np.uint8(sobel / sobel.max() * 255)
    
    # 3. Laplacian Edge Detection
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
    laplacian = np.uint8(np.absolute(laplacian))
    
    # 4. Scharr Edge Detection
    scharrx = cv2.Scharr(blurred, cv2.CV_64F, 1, 0)
    scharry = cv2.Scharr(blurred, cv2.CV_64F, 0, 1)
    scharr = np.sqrt(scharrx**2 + scharry**2)
    scharr = np.uint8(scharr / scharr.max() * 255)
    
    # Display all techniques
    cv2.imshow('Original', frame)
    cv2.imshow('Canny', canny)
    cv2.imshow('Sobel', sobel)
    cv2.imshow('Laplacian', laplacian)
    cv2.imshow('Scharr', scharr)
    
    print("Press any key to close all windows...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("=" * 60)
    print("Real-Time Edge Detection using OpenCV")
    print("=" * 60)
    
    # Run the main real-time edge detection
    main()
    
    # Uncomment the line below to see different edge detection techniques
    # demonstrate_edge_detection_techniques()
