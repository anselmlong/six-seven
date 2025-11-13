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

def display_picture(class_name):
    ref_dir = os.path.join(os.path.dirname(__file__), '..', 'public', 'reference-images')
    matches = glob(os.path.join(ref_dir, f"{class_name}.*"))
    file_path = matches[0] if matches else os.path.join(ref_dir, f"{class_name}.png")
    img = cv2.imread(file_path)
    if img is None:
        # no reference image found or failed to read; show placeholder or skip
        print(f"Reference image not found or unreadable: {file_path}")
        return
    # optional: resize reference to fit window (keep aspect)
    height, width = img.shape[:2]
    max_w, max_h = 320, 240
    if width > max_w or height > max_h:
        scale = min(max_w / width, max_h / height)
        img = cv2.resize(img, (int(width * scale), int(height * scale)))
    cv2.imshow("Reference Image", img)
 
# Function with signature: predictions, video_Frame
# import supervision to help visualize our predictions
from networkx import display
import supervision as sv

# create a bounding box annotator and label annotator to use in our custom sink
label_annotator = sv.LabelAnnotator()
box_annotator = sv.BoxAnnotator()

def my_custom_sink(predictions: dict, video_frame: VideoFrame):
    # get the supervision Detections object
    detections = predictions["predictions"]  # if this is the Detections object
    # if your structure is different adjust accordingly

    # extract class_name array
    class_names = []
    # Check if .data has the key
    if hasattr(detections, "data") and "class_name" in detections.data:
        class_names = list(detections.data["class_name"])
    else:
        # fallback: you might inspect detections.class_id or detections.data["class"]
        class_names = [""] * len(detections.xyxy)

    # Now for each detection call display_picture
    for name in class_names:
        if name:
            print("Detected class:", name)
            display_picture(name)

    # Then do your annotation / visualization
    # e.g. label the classes on the frame
    image = label_annotator.annotate(
        scene=video_frame.image.copy(),
        detections=detections,
        labels=class_names
    )
    image = box_annotator.annotate(image, detections=detections)
    cv2.imshow("Predictions", image)
    cv2.waitKey(1)


# 2. Initialize a pipeline object
pipeline = InferencePipeline.init_with_workflow(
    api_key=ROBOFLOW_API_KEY,
    workspace_name="fashion-vision",
    workflow_id="detect-and-classify",
    video_reference=0, # Path to video, device id (int, usually 0 for built in webcams), or RTSP stream url
    max_fps=30,
    on_prediction=my_custom_sink  # Use our custom sink function
)

# 3. Start the pipeline
print("Pipeline started. Press 'q' in the video window to exit.")
pipeline.start()
pipeline.join()
display_picture("thumbs-up")


"""
{'output_image': <inference.core.workflows.execution_engine.entities.base.WorkflowImageData object at 0x000002B3D053B950>, 
'predictions': Detections(
    xyxy=array([[     196.96,      185.77,      263.29,      339.19]]), 
	mask=None, 
	confidence=array([     0.4048]), 
	class_id=array([8]), 
	tracker_id=None, 
	data={
		'class_name': array(['thumbs-up'], dtype='<U9'), 
		'detection_id': array(['359e8a91-ff36-4bf7-87a
"""