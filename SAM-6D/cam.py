import pyzed.sl as sl
import cv2
import numpy as np
import json
import os
from datetime import datetime

def capture_zed_data(output_dir="Data/CHewing_Gum_Package", capture_name=None):
    """
    Capture RGB, depth, and camera intrinsics from ZED camera
    and save them in SAM-6D compatible format
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize ZED camera
    zed = sl.Camera()
    
    # Set configuration parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD2K
    init_params.camera_fps = 5
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL
    init_params.coordinate_units = sl.UNIT.MILLIMETER  # Important: use millimeters for depth
    
    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"Error opening ZED camera: {err}")
        return False
    
    # Get camera information
    camera_info = zed.get_camera_information()
    
    # Create sl.Mat objects to hold the images
    image_zed = sl.Mat()
    depth_zed = sl.Mat()
    
    # Runtime parameters
    runtime_params = sl.RuntimeParameters()
        
    print("ZED camera initialized successfully!")
    print("Press 'c' to capture, 'q' to quit")
    
    while True:
        # Grab an image
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            # Retrieve left image (RGB)
            zed.retrieve_image(image_zed, sl.VIEW.LEFT)
            
            # Retrieve depth map
            zed.retrieve_measure(depth_zed, sl.MEASURE.DEPTH)
            
            # Convert to numpy arrays
            rgb_image = image_zed.get_data()
            depth_image = depth_zed.get_data()
            
            # Convert RGB from BGRA to RGB (ZED returns BGRA)
            rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGRA2RGB)
            
            # Display the RGB image for preview
            display_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            cv2.imshow("ZED RGB (Press 'c' to capture, 'q' to quit)", display_image)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('c'):
                # Generate filename with timestamp if not provided
                if capture_name is None:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"capture_{timestamp}"
                else:
                    filename = capture_name
                
                # Save RGB image (convert RGB to BGR for OpenCV)
                rgb_path = os.path.join(output_dir, f"{filename}_rgb.png")
                rgb_bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
                success = cv2.imwrite(rgb_path, rgb_bgr)
                if not success:
                    print(f"Failed to save RGB image to {rgb_path}")
                    continue
                
                # Save depth image (in millimeters as uint16)
                depth_path = os.path.join(output_dir, f"{filename}_depth.png")
                
                # Handle NaN and infinite values in depth
                depth_clean = np.nan_to_num(depth_image, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Ensure depth is in the correct range and format
                depth_clean = np.clip(depth_clean, 0, 65535).astype(np.uint16)
                
                # Save depth as 16-bit PNG
                success = cv2.imwrite(depth_path, depth_clean)
                if not success:
                    print(f"Failed to save depth image to {depth_path}")
                    continue
                
                # Get camera intrinsics
                left_cam = camera_info.camera_configuration.calibration_parameters.left_cam
                
                # Create simplified camera information dictionary (only what SAM-6D needs)
                camera_data = {
                    "cam_K": [
                        left_cam.fx, 0.0, left_cam.cx,
                        0.0, left_cam.fy, left_cam.cy,
                        0.0, 0.0, 1.0
                    ],
                    "depth_scale": 1.0  # ZED outputs in millimeters, SAM-6D expects millimeters
                }
                
                # Save camera intrinsics
                camera_path = os.path.join(output_dir, f"{filename}_camera.json")
                with open(camera_path, 'w') as f:
                    json.dump(camera_data, f, indent=2)
                
                print(f"Captured data saved:")
                print(f"  RGB: {rgb_path}")
                print(f"  Depth: {depth_path}")
                print(f"  Camera: {camera_path}")
                print(f"  Image resolution: {rgb_image.shape[1]}x{rgb_image.shape[0]}")
                print(f"  Depth range: {np.min(depth_clean[depth_clean > 0]):.1f} - {np.max(depth_clean):.1f} mm")
                
                # Export paths for demo script
                print("\nTo use with SAM-6D demo, set these environment variables:")
                print(f"export RGB_PATH={os.path.abspath(rgb_path)}")
                print(f"export DEPTH_PATH={os.path.abspath(depth_path)}")
                print(f"export CAMERA_PATH={os.path.abspath(camera_path)}")
                
                break
                
            elif key == ord('q'):
                print("Capture cancelled")
                break
    
    # Clean up
    cv2.destroyAllWindows()
    zed.close()
    return True

def create_sam6d_compatible_files(output_dir="Data/Cone"):
    """
    Create symlinks with SAM-6D expected names (rgb.png, depth.png, camera.json)
    """
    # Find the most recent capture files
    if not os.path.exists(output_dir):
        print(f"Output directory {output_dir} does not exist.")
        return False
        
    files = os.listdir(output_dir)
    
    # Find RGB, depth, and camera files
    rgb_files = [f for f in files if f.endswith('_rgb.png')]
    depth_files = [f for f in files if f.endswith('_depth.png')]
    camera_files = [f for f in files if f.endswith('_camera.json')]
    
    if not rgb_files or not depth_files or not camera_files:
        print("No capture files found. Please run capture first.")
        return False
    
    # Use the most recent files (assuming timestamp naming)
    rgb_file = sorted(rgb_files)[-1]
    depth_file = sorted(depth_files)[-1]
    camera_file = sorted(camera_files)[-1]
    
    # Create symlinks or copies with expected names
    try:
        # Remove existing files if they exist
        for fname in ['rgb.png', 'depth.png', 'camera.json']:
            fpath = os.path.join(output_dir, fname)
            if os.path.exists(fpath):
                os.remove(fpath)
        
        # Create symlinks (or copies on Windows)
        try:
            os.symlink(rgb_file, os.path.join(output_dir, 'rgb.png'))
            os.symlink(depth_file, os.path.join(output_dir, 'depth.png'))
            os.symlink(camera_file, os.path.join(output_dir, 'camera.json'))
            
            print(f"Created SAM-6D compatible symlinks in {output_dir}:")
            print(f"  rgb.png -> {rgb_file}")
            print(f"  depth.png -> {depth_file}")
            print(f"  camera.json -> {camera_file}")
            
        except OSError:
            # Fallback to copying files if symlinks fail
            import shutil
            shutil.copy2(os.path.join(output_dir, rgb_file), os.path.join(output_dir, 'rgb.png'))
            shutil.copy2(os.path.join(output_dir, depth_file), os.path.join(output_dir, 'depth.png'))
            shutil.copy2(os.path.join(output_dir, camera_file), os.path.join(output_dir, 'camera.json'))
            
            print(f"Copied SAM-6D compatible files to {output_dir}")
        
        return True
        
    except Exception as e:
        print(f"Error creating compatible files: {e}")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Capture ZED camera data for SAM-6D")
    parser.add_argument("--output_dir", default="SAM-6D/Data/cube", help="Output directory for captured data")
    parser.add_argument("--name", help="Custom name for capture files")
    parser.add_argument("--create_links", action="store_true", help="Create SAM-6D compatible symlinks")
    
    args = parser.parse_args()
    
    if args.create_links:
        create_sam6d_compatible_files(args.output_dir)
    else:
        if capture_zed_data(args.output_dir, args.name):
            # Automatically create compatible files after capture
            create_sam6d_compatible_files(args.output_dir)
            
            # Print environment variables for easy copy-paste
            abs_output_dir = os.path.abspath(args.output_dir)
            print(f"\nEnvironment variables for SAM-6D demo:")
            print(f"export RGB_PATH={abs_output_dir}/rgb.png")
            print(f"export DEPTH_PATH={abs_output_dir}/depth.png")
            print(f"export CAMERA_PATH={abs_output_dir}/camera.json")
            print(f"export OUTPUT_DIR={abs_output_dir}/outputs")