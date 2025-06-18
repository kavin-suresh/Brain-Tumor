import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import networkx as nx
from scipy.spatial.distance import pdist, squareform
import os
import random

# Set random seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

class TumorDetectorApp:
    def __init__(self, root):
        """Initialize the TumorDetectorApp with a Tkinter window."""
        self.root = root
        self.root.title("Brain Tumor Detector with Similarity")
        self.root.geometry("800x600")  # Slightly larger window for better layout
        
        # Main container
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Control buttons frame
        self.button_frame = tk.Frame(self.main_frame)
        self.button_frame.pack(pady=5)
        
        self.browse_btn = tk.Button(self.button_frame, text="Select Images (1-10)", 
                                    command=self.load_images)
        self.browse_btn.pack(side=tk.LEFT, padx=5)
        
        self.process_btn = tk.Button(self.button_frame, text="Detect Tumors", 
                                     command=self.process_images)
        self.process_btn.pack(side=tk.LEFT, padx=5)
        
        self.similarity_btn = tk.Button(self.button_frame, text="Compute Similarities", 
                                        command=self.compute_similarities, state=tk.DISABLED)
        self.similarity_btn.pack(side=tk.LEFT, padx=5)
        
        self.heatmap_btn = tk.Button(self.button_frame, text="Show Heatmap", 
                                     command=self.show_heatmap, state=tk.DISABLED)
        self.heatmap_btn.pack(side=tk.LEFT, padx=5)
        
        self.graph_btn = tk.Button(self.button_frame, text="Show Graph", 
                                   command=self.show_graph, state=tk.DISABLED)
        self.graph_btn.pack(side=tk.LEFT, padx=5)
        
        self.counterfactual_btn = tk.Button(self.button_frame, text="Show Counterfactuals",
                                    command=self.show_counterfactuals, state=tk.DISABLED)
        self.counterfactual_btn.pack(side=tk.LEFT, padx=5)
        self.emotional_response_btn = tk.Button(self.button_frame, text="Show Emotional Response", 
                                    command=self.show_emotional_response, state=tk.DISABLED)
        self.emotional_response_btn.pack(side=tk.LEFT, padx=5)
        
        # Table for displaying images and similarities
        self.table_frame = tk.Frame(self.main_frame)
        self.table_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Inside _init_()
        # Headers
        tk.Label(self.table_frame, text="Original Image", width=20, 
                 font=('Arial', 10, 'bold')).grid(row=0, column=0, padx=5, pady=2)
        tk.Label(self.table_frame, text="Tumor Detection", width=20, 
                 font=('Arial', 10, 'bold')).grid(row=0, column=1, padx=5, pady=2)
        tk.Label(self.table_frame, text="Most Similar Tumor", width=20, 
                 font=('Arial', 10, 'bold')).grid(row=0, column=2, padx=5, pady=2)
        
        # Create 10 rows for images, results, and similarity labels
        self.image_labels = []
        self.result_labels = []
        self.most_similar_labels = []
        self.stress_level_label = []
        for i in range(10):
            img_label = tk.Label(self.table_frame, width=200, height=150, relief="groove", bg='white')
            img_label.grid(row=i+1, column=0, padx=5, pady=2, sticky='nsew')
            result_label = tk.Label(self.table_frame, width=200, height=150, relief="groove", bg='white')
            result_label.grid(row=i+1, column=1, padx=5, pady=2, sticky='nsew')
            most_similar_label = tk.Label(self.table_frame, width=20, relief="groove", text="", 
                                          anchor='center', bg='white')
            most_similar_label.grid(row=i+1, column=2, padx=5, pady=2, sticky='nsew')
            label = tk.Label(self.table_frame, text="Stress Level: ", font=('Arial', 10))
            label.grid(row=i+1, column=3, padx=5, pady=2, sticky='nsew')  # Add to column 3 (new column)
            
            self.image_labels.append(img_label)
            self.result_labels.append(result_label)
            self.most_similar_labels.append(most_similar_label)
            self.stress_level_label.append(label)
        
        # Configure grid weights for resizing
        for i in range(11):
            self.table_frame.rowconfigure(i, weight=1)
        for j in range(3):
            self.table_frame.columnconfigure(j, weight=1)
        
        # Initialize variables
        self.image_paths = []  # List to store image file paths
        self.tumor_features = []  # List to store tumor features
        self.valid_indices = []  # Indices of images with detected tumors
        self.similarity_matrix = None  # Store similarity matrix

    def load_images(self):
        """Load up to 10 images and display them in the GUI."""
        self.image_paths = []
        self.tumor_features = []
        self.similarity_matrix = None
        for label in self.image_labels + self.result_labels:
            label.configure(image='', text='')
        for label in self.most_similar_labels:
            label.configure(text='')
        self.similarity_btn.config(state=tk.DISABLED)
        self.heatmap_btn.config(state=tk.DISABLED)
        self.graph_btn.config(state=tk.DISABLED)
        
        paths = filedialog.askopenfilenames(
            title="Select Brain Images",
            filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp")],
            multiple=True
        )
        
        if not paths:
            return
        if len(paths) > 10:
            messagebox.showerror("Error", "Maximum 10 images allowed!")
            return
        
        self.image_paths = list(paths)[:10]
        
        for idx, path in enumerate(self.image_paths):
            try:
                img = Image.open(path)
                img.thumbnail((200, 150))  # Adjusted size for better visibility
                img_tk = ImageTk.PhotoImage(img)
                self.image_labels[idx].configure(image=img_tk)
                self.image_labels[idx].image = img_tk  # Keep reference
            except Exception as e:
                self.show_error(idx, f"Load Error: {str(e)}")

    def process_images(self):
        """Process loaded images to detect tumors and display results."""
        if not self.image_paths:
            messagebox.showwarning("Warning", "No images selected!")
            return
        
        self.tumor_features = []
        self.similarity_matrix = None
        for idx, path in enumerate(self.image_paths):
            try:
                result_img = self.detect_tumor(path)
                result_img.thumbnail((200, 150))
                img_tk = ImageTk.PhotoImage(result_img)
                self.result_labels[idx].configure(image=img_tk)
                self.result_labels[idx].image = img_tk
            except Exception as e:
                self.show_error(idx, f"Processing Error: {str(e)}")
                self.tumor_features.append(None)
        
        # Enable similarity button if at least 2 tumors are detected
        self.valid_indices = [idx for idx, f in enumerate(self.tumor_features) if f is not None]
        num_valid = len(self.valid_indices)
        if num_valid >= 2:
            self.similarity_btn.config(state=tk.NORMAL)
            messagebox.showinfo("Info", f"Processed {num_valid} images with tumors. Ready to compute similarities.")
        else:
            self.similarity_btn.config(state=tk.DISABLED)
            self.heatmap_btn.config(state=tk.DISABLED)
            self.graph_btn.config(state=tk.DISABLED)
            messagebox.showinfo("Info", f"Processed {num_valid} images with tumors. Need at least 2 for similarity computation.")

    def detect_tumor(self, path):
        """Detect tumor in an image and return the processed image."""
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError("Could not read image file")

        # Preprocessing
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)

        # Skull Stripping
        _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            self.tumor_features.append(None)
            return self.create_no_tumor_image(image)

        # Find brain contour closest to image center
        image_center = (image.shape[1] / 2, image.shape[0] / 2)
        min_distance = float('inf')
        brain_contour = None
        for cnt in contours:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                distance = ((cx - image_center[0])**2 + (cy - image_center[1])**2)**0.5
                if distance < min_distance:
                    min_distance = distance
                    brain_contour = cnt
        if brain_contour is None:
            self.tumor_features.append(None)
            return self.create_no_tumor_image(image)

        brain_mask = np.zeros_like(closed)
        cv2.drawContours(brain_mask, [brain_contour], -1, 255, -1)
        brain = cv2.bitwise_and(enhanced, enhanced, mask=brain_mask)

        # Tumor Detection
        brain_pixels = brain[brain_mask > 0]
        if brain_pixels.size == 0:
            self.tumor_features.append(None)
            return self.create_no_tumor_image(image)
        mean_intensity = np.mean(brain_pixels)
        std_intensity = np.std(brain_pixels)
        tumor_threshold = mean_intensity + 2 * std_intensity

        tumor_bin = (brain > tumor_threshold).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        tumor_bin = cv2.morphologyEx(tumor_bin, cv2.MORPH_OPEN, kernel)
        tumor_bin = cv2.morphologyEx(tumor_bin, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(tumor_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        color_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            cv2.rectangle(color_img, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Compute normalized tumor features
            center_x = x + w / 2
            center_y = y + h / 2
            norm_center_x = center_x / image.shape[1]
            norm_center_y = center_y / image.shape[0]
            norm_w = w / image.shape[1]
            norm_h = h / image.shape[0]
            self.tumor_features.append([norm_center_x, norm_center_y, norm_w, norm_h])
            return Image.fromarray(color_img)
        else:
            self.tumor_features.append(None)
            return self.create_no_tumor_image(image)

    def create_no_tumor_image(self, image):
        """Create an image indicating no tumor was detected."""
        color_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        cv2.putText(color_img, "No Tumor Detected", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return Image.fromarray(color_img)

    def show_error(self, idx, message):
        """Display an error message in the result label.""" 
        error_img = Image.new('RGB', (200, 150), color='red')
        draw = ImageDraw.Draw(error_img)
        draw.text((10, 60), message, fill='white')
        img_tk = ImageTk.PhotoImage(error_img)
        self.result_labels[idx].configure(image=img_tk)
        self.result_labels[idx].image = img_tk

    def compute_similarities(self):
        """Compute and display similarities between detected tumors.""" 
        if len(self.valid_indices) < 2:
            messagebox.showerror("Error", "Need at least 2 images with detected tumors.")
            return
        
        # Extract features for valid tumors
        features = np.array([self.tumor_features[idx] for idx in self.valid_indices])
        
        # Compute pairwise Euclidean distances
        distances = squareform(pdist(features, 'euclidean'))
        
        # Compute sigma for Gaussian kernel
        sigma = np.mean(distances) if np.mean(distances) > 0 else 1  # Avoid division by zero
        self.similarity_matrix = np.exp(-distances**2 / sigma**2) * 100  # Similarity in percentage
        
        # Display the most similar tumor for each
        for i in range(len(self.valid_indices)):
            similarities = self.similarity_matrix[i].copy()
            similarities[i] = -1  # Exclude self
            j = np.argmax(similarities)  # Index of most similar tumor
            sim = similarities[j]  # Similarity percentage
            text = f"Image {self.valid_indices[j]+1}, {sim:.2f}%"
            self.most_similar_labels[self.valid_indices[i]].configure(text=text)
        
        # Enable visualization buttons
        self.heatmap_btn.config(state=tk.NORMAL)
        self.graph_btn.config(state=tk.NORMAL)
        self.counterfactual_btn.config(state=tk.NORMAL)
        self.emotional_response_btn.config(state=tk.NORMAL)
        
    def show_heatmap(self):
        """Display a heatmap of the similarity matrix.""" 
        if self.similarity_matrix is None:
            messagebox.showerror("Error", "Similarity matrix not computed.")
            return
        
        top = tk.Toplevel(self.root)
        top.title("Similarity Heatmap")
        
        fig = plt.Figure(figsize=(6, 5))
        ax = fig.add_subplot(111)
        cax = ax.imshow(self.similarity_matrix, cmap='viridis', interpolation='nearest')
        fig.colorbar(cax, label='Similarity (%)')
        
        labels = [str(idx+1) for idx in self.valid_indices]
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Image")
        ax.set_ylabel("Image")
        ax.set_title("Tumor Similarity Heatmap")
        
        canvas = FigureCanvasTkAgg(fig, master=top)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def show_graph(self):
        """Display a graph visualization of tumor similarities.""" 
        if self.similarity_matrix is None:
            messagebox.showerror("Error", "Similarity matrix not computed.")
            return
        
        top = tk.Toplevel(self.root)
        top.title("Similarity Graph")
        
        fig = plt.Figure(figsize=(6, 5))
        ax = fig.add_subplot(111)
        
        G = nx.Graph()
        # Add nodes (image numbers)
        for idx in self.valid_indices:
            G.add_node(idx+1)
        
        # Add edges where similarity > 50%
        for i in range(len(self.valid_indices)):
            for j in range(i+1, len(self.valid_indices)):
                sim = self.similarity_matrix[i, j]
                if sim > 50:
                    G.add_edge(self.valid_indices[i]+1, self.valid_indices[j]+1, weight=sim)
        
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', 
                node_size=500, font_size=10, ax=ax)
        ax.set_title("Tumor Similarity Graph (Similarity > 50%)")
        
        canvas = FigureCanvasTkAgg(fig, master=top)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def show_counterfactuals(self):
        """Visualize original and counterfactual tumor boxes side-by-side.""" 
        if self.similarity_matrix is None:
            messagebox.showerror("Error", "Please compute similarities first.")
            return

        top = tk.Toplevel(self.root)
        top.title("Counterfactual Explanations")

        for i, idx in enumerate(self.valid_indices):
            original_path = self.image_paths[idx]
            original_img = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)
            original_img_color = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)

        # Original tumor feature
            orig_feat = np.array(self.tumor_features[idx])

        # Find most similar tumor
            similarities = self.similarity_matrix[i].copy()
            similarities[i] = -1
            j = np.argmax(similarities)
            match_feat = np.array(self.tumor_features[self.valid_indices[j]])

        # Compute counterfactual (morph toward similar tumor)
            alpha = 0.3
            counterfactual_feat = orig_feat + alpha * (match_feat - orig_feat)

        # Convert bounding boxes to pixel values
            h, w = original_img.shape
            cx, cy, tw, th = orig_feat
            x1 = int((cx - tw/2) * w)
            y1 = int((cy - th/2) * h)
            x2 = int((cx + tw/2) * w)
            y2 = int((cy + th/2) * h)
            cv2.rectangle(original_img_color, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cf_cx, cf_cy, cf_tw, cf_th = counterfactual_feat
            cf_x1 = int((cf_cx - cf_tw/2) * w)
            cf_y1 = int((cf_cy - cf_th/2) * h)
            cf_x2 = int((cf_cx + cf_tw/2) * w)
            cf_y2 = int((cf_cy + cf_th/2) * h)
            cv2.rectangle(original_img_color, (cf_x1, cf_y1), (cf_x2, cf_y2), (0, 0, 255), 2)

        # Labels
            cv2.putText(original_img_color, "Original", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(original_img_color, "Counterfactual", (cf_x1, cf_y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Display in new Tk window
            img_pil = Image.fromarray(original_img_color)
            img_tk = ImageTk.PhotoImage(img_pil)

            label = tk.Label(top, image=img_tk)
            label.image = img_tk
            label.pack(pady=5)
    def simulate_emotional_impact(self, tumor_features):
        """Simulate emotional response (stress/anxiety) based on tumor size and location.
        Returns a simulated stress level from 0 to 100.""" 
        stress_level = 0
    # Use tumor width and height as factors for emotional impact.
    # Larger tumors cause more stress/anxiety
        if tumor_features[2] > 0.4:  # If tumor width is large
            stress_level += 30
        if tumor_features[3] > 0.4:  # If tumor height is large
            stress_level += 40
    
    # Tumors in critical regions (for example, near the brainstem)
        if tumor_features[1] > 0.4:  # Simulating if the tumor is in a critical area
            stress_level += 30
    
    # Adjust stress level based on general guidelines.
        stress_level = min(stress_level, 100)  # Cap the stress level at 100%
    
        return stress_level

    def show_emotional_response(self):
        """Display the emotional response (stress level) for each tumor.""" 
        if self.tumor_features is None:
            messagebox.showerror("Error", "No tumor features available.")
            return

    # Iterate over all detected tumors and simulate emotional impact
        for idx, feature in enumerate(self.tumor_features):
            if feature:  # Only process valid features
                stress_level = self.simulate_emotional_impact(feature)
            
            # Update the GUI with the calculated stress level
                self.stress_level_label[idx].configure(text=f"Stress Level: {stress_level}%")

if __name__ == "__main__":
    root = tk.Tk()
    app = TumorDetectorApp(root)
    root.mainloop()
