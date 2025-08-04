import airsim
import numpy as np
import cv2
import os

# Connect to AirSim
client = airsim.MultirotorClient()
client.confirmConnection()

# Create output folder if it doesn't exist
os.makedirs("airsim_photos", exist_ok=True)

# Request images from both fpv and tpv cameras
responses = client.simGetImages([
    # FPV
    airsim.ImageRequest("fpv", airsim.ImageType.Scene, False, False),
    airsim.ImageRequest("fpv", airsim.ImageType.Segmentation, False, False),
    airsim.ImageRequest("fpv", airsim.ImageType.DepthPerspective, True, False),
    # TPV
    airsim.ImageRequest("tpv", airsim.ImageType.Scene, False, False),
    airsim.ImageRequest("tpv", airsim.ImageType.Segmentation, False, False),
    airsim.ImageRequest("tpv", airsim.ImageType.DepthPerspective, True, False)
])

# Function to save either Scene/Segmentation or Depth images
def save_image(response, filename):
    if response.pixels_as_float:
        # Depth: grayscale from float values
        img1d = np.array(response.image_data_float, dtype=np.float32)
        img1d = np.clip(img1d, 0, 100)
        img1d = 255 * (img1d / 100)
        img2d = img1d.reshape(response.height, response.width).astype(np.uint8)
        cv2.imwrite(os.path.join("airsim_photos", filename), img2d)
    else:
        # Scene/Segmentation: RGB
        img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
        img_rgb = img1d.reshape(response.height, response.width, 3)
        cv2.imwrite(os.path.join("airsim_photos", filename), img_rgb)

# Save all images with consistent naming
save_image(responses[0], "fpv_scene.png")
save_image(responses[1], "fpv_segmentation.png")
save_image(responses[2], "fpv_depth.png")
save_image(responses[3], "tpv_scene.png")
save_image(responses[4], "tpv_segmentation.png")
save_image(responses[5], "tpv_depth.png")

print("Photos saved in airsim_photos/:")
print(" - fpv_scene.png, fpv_segmentation.png, fpv_depth.png")
print(" - tpv_scene.png, tpv_segmentation.png, tpv_depth.png")