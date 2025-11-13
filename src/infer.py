# 1. Import the InferencePipeline library
import re
from inference import InferencePipeline
# Import the built in render_boxes sink for visualizing results
from inference.core.interfaces.stream.sinks import render_boxes
import cv2
import os
from dotenv import load_dotenv
from glob import glob
from inference.core.interfaces.camera.entities import VideoFrame

# Load environment variables from .env file
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")  # Ensure the API key is set in your environment variables

# Cache for reference images to avoid repeated file I/O
reference_image_cache = {}
last_displayed_class = None
frame_counter = 0
FRAME_SKIP = 1  # Process every Nth frame (1=all, 2=every other, 3=every third, etc.)

def display_picture(class_name):
    global last_displayed_class
    
    # Only update if class changed (avoid redundant display calls)
    if class_name == last_displayed_class:
        return
    
    last_displayed_class = class_name
    
    # Check cache first
    if class_name in reference_image_cache:
        img = reference_image_cache[class_name]
        if img is not None:
            cv2.imshow("Reference Image", img)
        return
    
    # Load and cache
    ref_dir = os.path.join(os.path.dirname(__file__), '..', 'public', 'reference-images')
    matches = glob(os.path.join(ref_dir, f"{class_name}.*"))
    file_path = matches[0] if matches else os.path.join(ref_dir, f"{class_name}.png")
    img = cv2.imread(file_path)
    
    if img is None:
        reference_image_cache[class_name] = None
        return
    
    # Resize once and cache
    height, width = img.shape[:2]
    max_w, max_h = 320, 240
    if width > max_w or height > max_h:
        scale = min(max_w / width, max_h / height)
        img = cv2.resize(img, (int(width * scale), int(height * scale)))
    
    reference_image_cache[class_name] = img
    cv2.imshow("Reference Image", img)
 
# Function with signature: predictions, video_Frame
# import supervision to help visualize our predictions
from networkx import display
import supervision as sv

# create a bounding box annotator and label annotator to use in our custom sink

def my_custom_sink(predictions: dict, video_frame: VideoFrame):
    global frame_counter
    frame_counter += 1
    
    # Still display the frame for smooth preview
    if predictions.get("output_image"): # Display an image from the workflow response
        cv2.imshow("Workflow Image", predictions["output_image"].numpy_image)

    # get the supervision Detections object
    detections = predictions["predictions"]

    # extract class_name array
    class_names = []

    if hasattr(detections, "data") and "class_name" in detections.data:
        class_names = list(detections.data["class_name"])
    else:
        class_names = [""] * len(detections.xyxy)

    # Only show reference for first detection (not all detections)
    if class_names and class_names[0]:
        print ("Displaying reference for:", class_names[0])
        display_picture(class_names[0])

    if cv2.waitKey(1) & 0xFF == ord('q'):
        pipeline.terminate()
        cv2.destroyAllWindows()


# 2. Initialize a pipeline object
pipeline = InferencePipeline.init_with_workflow(
    api_key=ROBOFLOW_API_KEY,
    workspace_name="fashion-vision",
    workflow_id="detect-and-classify-3",
    video_reference=0, # Path to video, device id (int, usually 0 for built in webcams), or RTSP stream url
    max_fps=30,  # Reduced from 60 for better stability
    on_prediction=my_custom_sink  # Use our custom sink function
)

# 3. Start the pipeline
print("Pipeline started. Press 'q' in the video window to exit.")
pipeline.start()
pipeline.join()
