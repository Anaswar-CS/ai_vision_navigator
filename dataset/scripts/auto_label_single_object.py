import argparse
import os
import cv2
import numpy as np
from pathlib import Path
import shutil

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', required=True, type=str)
    parser.add_argument('--class-id', required=True, type=int)
    parser.add_argument('--out', required=True, type=str)
    parser.add_argument('--debug', action='store_true')
    return parser.parse_args()

def main():
    args = parse_args()
    img_dir = Path(args.images)
    out_dir = Path(args.out)
    
    lbl_dir = out_dir / 'labels'
    img_out_dir = out_dir / 'images'
    debug_dir = out_dir / 'debug'
    
    lbl_dir.mkdir(parents=True, exist_ok=True)
    img_out_dir.mkdir(parents=True, exist_ok=True)
    if args.debug:
        debug_dir.mkdir(parents=True, exist_ok=True)
        
    for img_path in list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png')):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
            
        # Basic auto-labeling: find largest contour via edge detection/thresholding
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        
        # Try a simple threshold or adaptive
        # _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            continue
            
        # Assume largest contour is the object
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        
        # YOLO format
        ih, iw = img.shape[:2]
        cx = (x + w/2) / iw
        cy = (y + h/2) / ih
        nw = w / iw
        nh = h / ih
        
        # Save label
        lbl_path = lbl_dir / (img_path.stem + '.txt')
        with open(lbl_path, 'w') as f:
            f.write(f'{args.class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n')
            
        # Copy image
        shutil.copy(str(img_path), str(img_out_dir / img_path.name))
        
        if args.debug:
            debug_img = img.copy()
            cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.imwrite(str(debug_dir / img_path.name), debug_img)
            
    print(f'Processed images and saved to {out_dir}')

if __name__ == '__main__':
    main()
