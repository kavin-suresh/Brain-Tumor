import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from skimage.segmentation import slic
from skimage import graph
from skimage.measure import regionprops

# Step 1: User selects an image file
root = tk.Tk()
root.withdraw()  # Hide the root window
file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp")])
if not file_path:
    print("No file selected.")
    exit()

# Step 2: Load the image in grayscale
image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
if image is None:
    print("Error: Could not load image. Please check the file path.")
    exit()

# Step 3: Preprocessing - Noise reduction with Gaussian blur
blurred = cv2.GaussianBlur(image, (5, 5), 0)

# Step 4: Preprocessing - Contrast enhancement with CLAHE
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
enhanced = clahe.apply(blurred)

# Step 5: Skull stripping - Binarize using Otsu’s thresholding
_, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Step 6: Find the largest connected component (brain region)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
brain_contour = max(contours, key=cv2.contourArea)
otsu_thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
_, brain_mask = cv2.threshold(enhanced, otsu_thresh * 1.1, 255, cv2.THRESH_BINARY)  # Adjust multiplier
cv2.drawContours(brain_mask, [brain_contour], -1, 255, -1)

# Step 7: Isolate the brain region
brain = cv2.bitwise_and(enhanced, enhanced, mask=brain_mask)

# Step 8: Initial tumor segmentation - Threshold based on intensity
brain_pixels = brain[brain_mask > 0]
mean_intensity = np.mean(brain_pixels)
std_intensity = np.std(brain_pixels)
tumor_threshold = mean_intensity + 3 * std_intensity  # Adjusted to 2 for less strict detection
tumor_bin = (brain > tumor_threshold).astype(np.uint8) * 255
print("Tumor threshold:", tumor_threshold)
print("Number of tumor pixels initially detected:", np.sum(tumor_bin > 0))

# Step 9: Clean up the binary image with morphological operations
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))  # Can adjust to (3, 3) or remove if needed
tumor_bin = cv2.morphologyEx(tumor_bin, cv2.MORPH_OPEN, kernel)  # Remove noise
tumor_bin = cv2.morphologyEx(tumor_bin, cv2.MORPH_CLOSE, kernel)  # Fill holes
tumor_bin = cv2.erode(tumor_bin, kernel, iterations=1)
print("Number of tumor pixels after morphological operations:", np.sum(tumor_bin > 0))


# Step 10: Generate superpixels for refinement
superpixels = slic(brain, n_segments=500, compactness=10, sigma=1, channel_axis=None)

min_label = np.min(superpixels)
print("Minimum superpixel label:", min_label)

# Step 11: Construct a Region Adjacency Graph (RAG)
rag = graph.rag_mean_color(brain, superpixels)

# Step 12: Extract features for each superpixel
props = regionprops(superpixels, intensity_image=brain)
features = []
for prop in props:
    mean_intensity = prop.mean_intensity
    std_intensity = np.std(brain[prop.coords[:, 0], prop.coords[:, 1]])
    initial_label = 1 if np.mean(tumor_bin[prop.coords[:, 0], prop.coords[:, 1]] > 0) > 0.5 else 0
    features.append([mean_intensity, std_intensity, initial_label])
features = np.array(features)

# Step 13: Refine labels using majority voting on the RAG
initial_labels = features[:, 2].astype(int)
# Refinement with majority voting
refined_labels = np.zeros_like(initial_labels)
# Modified refinement logic
for node in rag.nodes():
    neighbors = list(rag.neighbors(node))
    if neighbors:
        neighbor_labels = [initial_labels[n - 1] for n in neighbors]
        if initial_labels[node - 1] == 1 and any(neighbor_labels)>=2:
            refined_labels[node - 1] = 1  # Keep tumor label if any neighbor is a tumor
        else:
            majority_label = 1 if sum(neighbor_labels) > len(neighbor_labels) / 2 else 0
            refined_labels[node - 1] = majority_label
    else:
        refined_labels[node - 1] = 0  #initial_labels[node - 1]

# Step 14: Map refined labels back to pixel-level mask
refined_mask = np.zeros_like(tumor_bin)
for i, prop in enumerate(props):
    refined_mask[prop.coords[:, 0], prop.coords[:, 1]] = refined_labels[i]


print("Initial tumor superpixels:", np.sum(initial_labels))
print("Refined tumor superpixels:", np.sum(refined_labels))
print("Tumor pixels in final mask:", np.sum(refined_mask > 0))
# Step 15: Visualize the refined segmentation
overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
overlay[refined_mask > 0] = (0, 255, 0)  # Green for tumor regions
cv2.imshow('Refined Tumor Segmentation', overlay)
cv2.waitKey(0)
cv2.destroyAllWindows()