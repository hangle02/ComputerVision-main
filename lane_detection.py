import cv2
import numpy as np
import os
import glob

def detect_full_road(img):
    height, width = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    mask = np.zeros_like(edges)
    cv2.rectangle(mask, (0, int(height * 0.55)), (width, height), 255, -1)
    masked_edges = cv2.bitwise_and(edges, mask)

    # 2. Hough Transform
    lines = cv2.HoughLinesP(
        masked_edges, 
        rho = 1, 
        theta=np.pi/180, 
        threshold=40, 
        minLineLength=40, 
        maxLineGap=100
    )

    left_road_edges = []
    right_road_edges = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 == x1: continue
                
            slope = (y2 - y1) / (x2 - x1)
            if abs(slope) < 0.3 or abs(slope) > 5:
                continue
                
            intercept = y1 - slope * x1
            x_bottom = (height - intercept) / slope
            
            if slope < 0:
                left_road_edges.append((x_bottom, slope, intercept))
            else:
                right_road_edges.append((x_bottom, slope, intercept))

    if len(left_road_edges) > 0 and len(right_road_edges) > 0:
   
        left_road_edges.sort(key=lambda x: x[0])
        _, m_left, b_left = left_road_edges[0]
        
        
        right_road_edges.sort(key=lambda x: x[0], reverse=True)
        _, m_right, b_right = right_road_edges[0]
        
        
        y1_draw = height
        y2_draw = int(height * 0.6)
        
        x1_left = int((y1_draw - b_left) / m_left)
        x2_left = int((y2_draw - b_left) / m_left)
        cv2.line(img, (x1_left, y1_draw), (x2_left, y2_draw), (0, 255, 0), 4)
        
        x1_right = int((y1_draw - b_right) / m_right)
        x2_right = int((y2_draw - b_right) / m_right)
        cv2.line(img, (x1_right, y1_draw), (x2_right, y2_draw), (255, 0, 0), 4)
        
        # mark center
        target_y = int(height * 0.8) 
        x_left_edge = int((target_y - b_left) / m_left)
        x_right_edge = int((target_y - b_right) / m_right)
        
        mid_x = int((x_left_edge + x_right_edge) / 2)
        
        cv2.circle(img, (mid_x, target_y), 10, (0, 0, 255))
        #cv2.putText(img, "Road Center", (mid_x - 80, target_y - 20), 
         #           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return img

def process_folder(input_folder, output_folder):
    
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

   
    image_paths = glob.glob(os.path.join(input_folder, '*.*'))
    
    
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_paths = [p for p in image_paths if p.lower().endswith(valid_extensions)]

    if len(image_paths) == 0:
        print(f"The '{input_folder}' is empty")
        return

    print(f"Processing {len(image_paths)}")

   
    for i, img_path in enumerate(image_paths):
        
        img = cv2.imread(img_path)
        if img is None:
            print(f"Image reading error: {img_path}")
            continue
            
        
        processed_img = detect_full_road(img)
        
        
        filename = os.path.basename(img_path)
        output_path = os.path.join(output_folder, f"out_{filename}")
        
        
        cv2.imwrite(output_path, processed_img)
        print(f"[{i+1}/{len(image_paths)}] Saved: {output_path}")

    print("Complete! Check the output_images folder")


if __name__ == '__main__':
    
    INPUT_DIR = 'input_images'
    
    OUTPUT_DIR = 'output_images'

    process_folder(INPUT_DIR, OUTPUT_DIR)