import cv
import numpy as np
import networkx as nx
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import os

class TumorDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Brain Tumor Detection Pipeline")
        self.images = []
        self.tumors = []  # List of tumor features: [x, y, w, h]
        self.similarity_matrix = None
        self.stress_scores = []
        self.setup_gui()

    def setup_gui(self):
        # File selection
        self.load_btn = ttk.Button(self.root, text="Load MRI Images", command=self.load_images)
        self.load_btn.pack(pady=5)

        # Process buttons
        self.process_btn = ttk.Button(self.root, text="Process Images", command=self.run_pipeline, state='disabled')
        self.process_btn.pack(pady=5)

        # Display canvas
        self.canvas = tk.Canvas(self.root, width=512, height=512)
        self.canvas.pack(pady=5)

        # Results display
        self.result_label = ttk.Label(self.root, text="Results: Not processed")
        self.result_label.pack(pady=5)

    def load_images(self):
        # Load up to 10 grayscale MRI images
        files = filedialog.askopenfilenames(filetypes=[("Image files", "*.png *.jpg")])
        self.images = []
        for file in files[:10]:
            img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
            if img is not None and img.shape == (512, 512):
                self.images.append(img)
        if self.images:
            self.process_btn['state'] = 'normal'
            self.result_label.config(text=f"Loaded {len(self.images)} images")
        else:
            self.result_label.config(text="No valid 512x512 images loaded")

    def preprocess_image(self, img):
        # Skull-stripping: Isolate brain tissue
        _, thresh = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(img)
        cv2.drawContours(mask, contours, -1, 255, -1)
        brain_img = cv2.bitwise_and(img, mask)
        return brain_img

    def detect_tumors(self, img):
        # Detect bright tumor regions
        _, thresh = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        tumors = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w > 10 and h > 10:  # Filter small noise
                # Normalize features
                tumors.append([x/512, y/512, w/512, h/512])
        return tumors

    def compute_similarity(self, tumors):
        # Build GNN-like graph
        G = nx.complete_graph(len(tumors))
        weights = {}
        for i in range(len(tumors)):
            for j in range(i + 1, len(tumors)):
                # Inverse distance weight
                dist = np.sqrt((tumors[i][0] - tumors[j][0])**2 + (tumors[i][1] - tumors[j][1])**2)
                weights[(i, j)] = 1 / (dist + 0.01)  # Avoid division by zero

        # Normalize weights
        total_weight = sum(weights.values())
        for edge in weights:
            weights[edge] /= total_weight

        # Update features with weighted aggregation
        new_features = []
        for i in range(len(tumors)):
            updated = np.zeros(4)
            for j in range(len(tumors)):
                if i != j:
                    w = weights.get((min(i, j), max(i, j)), weights.get((max(i, j), min(i, j)), 0))
                    updated += w * np.array(tumors[j])
            new_features.append(updated.tolist())

        # Compute similarity matrix
        sim_matrix = np.zeros((len(tumors), len(tumors)))
        for i in range(len(tumors)):
            for j in range(i + 1, len(tumors)):
                diff = np.linalg.norm(np.array(new_features[i]) - np.array(new_features[j]))
                sim_matrix[i, j] = sim_matrix[j, i] = 100 * (1 - diff / 2)  # Scale to 0-100
        return sim_matrix, new_features

    def compute_counterfactual(self, tumors, sim_matrix):
        # Select high-similarity pair
        if len(tumors) < 2:
            return None
        i, j = np.unravel_index(np.argmax(sim_matrix), sim_matrix.shape)
        if i == j:
            return None
        # Interpolate features
        alpha = 0.5
        cf_features = alpha * np.array(tumors[i]) + (1 - alpha) * np.array(tumors[j])
        return cf_features.tolist()

    def compute_stress(self, tumor):
        # Heuristic: Stress based on size and centrality
        x, y, w, h = tumor
        size_score = 50 * (w * h)  # Area-based, max 50
        centrality = 1 - (abs(x - 0.5) + abs(y - 0.5))  # Closer to center (0.5, 0.5) is worse
        centrality_score = 50 * centrality
        return min(size_score + centrality_score, 100)  # Cap at 100

    def run_pipeline(self):
        if not self.images:
            self.result_label.config(text="No images loaded")
            return
        self.tumors = []
        self.stress_scores = []
        all_tumors = []

        # Process each image
        for img in self.images:
            brain_img = self.preprocess_image(img)
            tumors = self.detect_tumors(brain_img)
            all_tumors.extend(tumors)
            for tumor in tumors:
                self.stress_scores.append(self.compute_stress(tumor))
        self.tumors = all_tumors

        # Compute similarity and counterfactuals
        if len(self.tumors) > 0:
            self.similarity_matrix, updated_features = self.compute_similarity(self.tumors)
            cf_tumor = self.compute_counterfactual(self.tumors, self.similarity_matrix)
            self.display_results(self.images[0], self.tumors, cf_tumor)
            avg_stress = np.mean(self.stress_scores) if self.stress_scores else 0
            self.result_label.config(text=f"Tumors: {len(self.tumors)}, Avg Stress: {avg_stress:.1f}")
        else:
            self.result_label.config(text="No tumors detected")

    def display_results(self, img, tumors, cf_tumor):
        # Display image with tumor and counterfactual boxes
        display_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        for tumor in tumors:
            x, y, w, h = [int(v * 512) for v in tumor]
            cv2.rectangle(display_img, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green for real
        if cf_tumor:
            x, y, w, h = [int(v * 512) for v in cf_tumor]
            cv2.rectangle(display_img, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Red for counterfactual
        img_pil = Image.fromarray(display_img)
        img_tk = ImageTk.PhotoImage(img_pil)
        self.canvas.create_image(0, 0, anchor="nw", image=img_tk)
        self.canvas.image = img_tk

if __name__ == "__main__":
    root = tk.Tk()
    app = TumorDetectorApp(root)
    root.mainloop()import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import networkx as nx
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
        self.root.title("Brain Tumor Detector")
        self.root.geometry("900x700")
        
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.button_frame = tk.Frame(self.main_frame)
        self.button_frame.pack(pady=5)
        
        tk.Button(self.button_frame, text="Select Images (1-10)", command=self.load_images).pack(side=tk.LEFT, padx=5)
        tk.Button(self.button_frame, text="Detect Tumors", command=self.process_images).pack(side=tk.LEFT, padx=5)
        self.similarity_btn = tk.Button(self.button_frame, text="Compute Similarities", 
                                        command=self.compute_similarities, state=tk.DISABLED)
        self.similarity_btn.pack(side=tk.LEFT, padx=5)
        self.heatmap_btn = tk.Button(self.button_frame, text="Show Heatmap", 
                                     command=self.show_heatmap, state=tk.DISABLED)
        self.heatmap_btn.pack(side=tk.LEFT, padx=5)
        self.graph_btn = tk.Button(self.button_frame, text="Show Graph", 
                                   command=self.show_graph, state=tk.DISABLED)
        self.graph_btn.pack(side=tk.LEFT, padx=5)
        self.cf_btn = tk.Button(self.button_frame, text="Show Counterfactuals",
                                command=self.show_counterfactuals, state=tk.DISABLED)
        self.cf_btn.pack(side=tk.LEFT, padx=5)
        self.stress_btn = tk.Button(self.button_frame, text="Show Stress Levels", 
                                    command=self.show_emotional_response, state=tk.DISABLED)
        self.stress_btn.pack(side=tk.LEFT, padx=5)
        
        self.table_frame = tk.Frame(self.main_frame)
        self.table_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        tk.Label(self.table_frame, text="Original Image", width=20, font=('Arial', 10, 'bold')).grid(row=0, column=0, padx=5, pady=2)
        tk.Label(self.table_frame, text="Tumor Detection", width=20, font=('Arial', 10, 'bold')).grid(row=0, column=1, padx=5, pady=2)
        tk.Label(self.table_frame, text="Stress Level", width=20, font=('Arial', 10, 'bold')).grid(row=0, column=2, padx=5, pady=2)
        
        self.image_labels = []
        self.result_labels = []
        self.stress_labels = []
        for i in range(10):
            img_label = tk.Label(self.table_frame, width=200, height=150, relief="groove", bg='white')
            img_label.grid(row=i+1, column=0, padx=5, pady=2, sticky='nsew')
            result_label = tk.Label(self.table_frame, width=200, height=150, relief="groove", bg='white')
            result_label.grid(row=i+1, column=1, padx=5, pady=2, sticky='nsew')
            stress_label = tk.Label(self.table_frame, text="Stress: N/A", width=20, relief="groove", anchor='center', bg='white')
            stress_label.grid(row=i+1, column=2, padx=5, pady=2, sticky='nsew')
            self.image_labels.append(img_label)
            self.result_labels.append(result_label)
            self.stress_labels.append(stress_label)
        
        for i in range(11):
            self.table_frame.rowconfigure(i, weight=1)
        for j in range(3):
            self.table_frame.columnconfigure(j, weight=1)
        
        self.image_paths = []
        self.tumor_features = []
        self.valid_indices = []
        self.similarity_matrix = None

    def load_images(self):
        """Load up to 10 images and display them."""
        self.image_paths = []
        self.tumor_features = []
        self.similarity_matrix = None
        for label in self.image_labels + self.result_labels + self.stress_labels:
            label.configure(image='', text='Stress: N/A')
        self.similarity_btn.config(state=tk.DISABLED)
        self.heatmap_btn.config(state=tk.DISABLED)
        self.graph_btn.config(state=tk.DISABLED)
        self.cf_btn.config(state=tk.DISABLED)
        self.stress_btn.config(state=tk.DISABLED)
        
        paths = filedialog.askopenfilenames(filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp")])
        if not paths:
            return
        if len(paths) > 10:
            messagebox.showerror("Error", "Maximum 10 images allowed!")
            return
        
        self.image_paths = list(paths)[:10]
        for idx, path in enumerate(self.image_paths):
            try:
                img = Image.open(path)
                img.thumbnail((200, 150))
                img_tk = ImageTk.PhotoImage(img)
                self.image_labels[idx].configure(image=img_tk)
                self.image_labels[idx].image = img_tk
            except Exception as e:
                self.show_error(idx, f"Load Error: {e}")

    def process_images(self):
        """Process images to detect tumors and compute stress."""
        if not self.image_paths:
            messagebox.showwarning("Warning", "No images selected!")
            return
        
        self.tumor_features = []
        self.similarity_matrix = None
        for idx, path in enumerate(self.image_paths):
            try:
                result_img, stress = self.detect_tumor(path)
                result_img.thumbnail((200, 150))
                img_tk = ImageTk.PhotoImage(result_img)
                self.result_labels[idx].configure(image=img_tk)
                self.result_labels[idx].image = img_tk
                self.stress_labels[idx].configure(text=f"Stress: {stress:.1f}%")
            except Exception as e:
                self.show_error(idx, f"Processing Error: {e}")
                self.tumor_features.append(None)
                self.stress_labels[idx].configure(text="Stress: N/A")
        
        self.valid_indices = [idx for idx, f in enumerate(self.tumor_features) if f is not None]
        if len(self.valid_indices) >= 2:
            self.similarity_btn.config(state=tk.NORMAL)
            messagebox.showinfo("Info", f"Processed {len(self.valid_indices)} images with tumors.")
        else:
            self.similarity_btn.config(state=tk.DISABLED)
            self.heatmap_btn.config(state=tk.DISABLED)
            self.graph_btn.config(state=tk.DISABLED)
            self.cf_btn.config(state=tk.DISABLED)
            self.stress_btn.config(state=tk.DISABLED)
            messagebox.showinfo("Info", f"Need 2+ tumors for similarity.")

    def detect_tumor(self, path):
        """Detect tumor, resize image, and compute stress."""
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError("Could not read image")
        
        # Resize to 512x512
        image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)
        
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
            return self.create_no_tumor_image(image), 0
        
        image_center = (image.shape[1] / 2, image.shape[0] / 2)
        brain_contour = min(contours, key=lambda cnt: (
            (cv2.moments(cnt)["m10"]/cv2.moments(cnt)["m00"] - image_center[0])**2 +
            (cv2.moments(cnt)["m01"]/cv2.moments(cnt)["m00"] - image_center[1])**2
        ) if cv2.moments(cnt)["m00"] != 0 else float('inf'))
        
        brain_mask = np.zeros_like(closed)
        cv2.drawContours(brain_mask, [brain_contour], -1, 255, -1)
        brain = cv2.bitwise_and(enhanced, enhanced, mask=brain_mask)
        
        # Tumor Detection
        brain_pixels = brain[brain_mask > 0]
        if brain_pixels.size == 0:
            return self.create_no_tumor_image(image), 0
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
            
            norm_center_x = (x + w/2) / 512
            norm_center_y = (y + h/2) / 512
            norm_w = w / 512
            norm_h = h / 512
            features = [norm_center_x, norm_center_y, norm_w, norm_h]
            self.tumor_features.append(features)
            
            # Compute stress
            stress = self.compute_stress(features)
            return Image.fromarray(color_img), stress
        else:
            self.tumor_features.append(None)
            return self.create_no_tumor_image(image), 0

    def create_no_tumor_image(self, image):
        """Create image indicating no tumor."""
        color_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        cv2.putText(color_img, "No Tumor Detected", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return Image.fromarray(color_img)

    def show_error(self, idx, message):
        """Display error in result label."""
        error_img = Image.new('RGB', (200, 150), color='red')
        draw = ImageDraw.Draw(error_img)
        draw.text((10, 60), message, fill='white')
        img_tk = ImageTk.PhotoImage(error_img)
        self.result_labels[idx].configure(image=img_tk)
        self.result_labels[idx].image = img_tk

    def compute_similarities(self):
        """Compute GNN-based similarities."""
        if len(self.valid_indices) < 2:
            messagebox.showerror("Error", "Need 2+ tumors.")
            return
        
        features = [self.tumor_features[idx] for idx in self.valid_indices]
        G = nx.complete_graph(len(features))
        weights = {}
        for i in range(len(features)):
            for j in range(i + 1, len(features)):
                dist = np.sqrt((features[i][0] - features[j][0])**2 + (features[i][1] - features[j][1])**2)
                weights[(i, j)] = 1 / (dist + 0.01)
        
        total_weight = sum(weights.values())
        for edge in weights:
            weights[edge] /= total_weight
        
        new_features = []
        for i in range(len(features)):
            updated = np.zeros(4)
            for j in range(len(features)):
                if i != j:
                    w = weights.get((min(i, j), max(i, j)), weights.get((max(i, j), min(i, j)), 0))
                    updated += w * np.array(features[j])
            new_features.append(updated.tolist())
        
        self.similarity_matrix = np.zeros((len(features), len(features)))
        for i in range(len(features)):
            for j in range(i + 1, len(features)):
                diff = np.linalg.norm(np.array(new_features[i]) - np.array(new_features[j]))
                sim = 100 * (1 - diff / 2)
                self.similarity_matrix[i, j] = self.similarity_matrix[j, i] = sim
        
        for i, idx in enumerate(self.valid_indices):
            similarities = self.similarity_matrix[i].copy()
            similarities[i] = -1
            j = np.argmax(similarities)
            sim = similarities[j]
            self.result_labels[idx].configure(text=f"Similar to Image {self.valid_indices[j]+1}: {sim:.1f}%")
        
        self.heatmap_btn.config(state=tk.NORMAL)
        self.graph_btn.config(state=tk.NORMAL)
        self.cf_btn.config(state=tk.NORMAL)
        self.stress_btn.config(state=tk.NORMAL)

    def compute_stress(self, tumor):
        """Compute stress based on size and centrality."""
        x, y, w, h = tumor
        size_score = 50 * (w * h)
        centrality = 1 - (abs(x - 0.5) + abs(y - 0.5))
        centrality_score = 50 * centrality
        return min(size_score + centrality_score, 100)

    def show_heatmap(self):
        """Display similarity heatmap."""
        if self.similarity_matrix is None:
            messagebox.showerror("Error", "No similarity matrix.")
            return
        
        top = tk.Toplevel(self.root)
        top.title("Similarity Heatmap")
        
        fig = plt.Figure(figsize=(5, 4))
        ax = fig.add_subplot(111)
        cax = ax.imshow(self.similarity_matrix, cmap='viridis')
        fig.colorbar(cax, label='Similarity (%)')
        
        labels = [str(idx+1) for idx in self.valid_indices]
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Image")
        ax.set_ylabel("Image")
        
        canvas = FigureCanvasTkAgg(fig, master=top)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def show_graph(self):
        """Display similarity graph."""
        if self.similarity_matrix is None:
            messagebox.showerror("Error", "No similarity matrix.")
            return
        
        top = tk.Toplevel(self.root)
        top.title("Similarity Graph")
        
        fig = plt.Figure(figsize=(5, 4))
        ax = fig.add_subplot(111)
        
        G = nx.Graph()
        for idx in self.valid_indices:
            G.add_node(idx+1)
        
        for i in range(len(self.valid_indices)):
            for j in range(i+1, len(self.valid_indices)):
                sim = self.similarity_matrix[i, j]
                if sim > 50:
                    G.add_edge(self.valid_indices[i]+1, self.valid_indices[j]+1, weight=sim)
        
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', 
                node_size=500, font_size=10, ax=ax)
        
        canvas = FigureCanvasTkAgg(fig, master=top)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def show_counterfactuals(self):
        """Show counterfactual tumor boxes."""
        if self.similarity_matrix is None:
            messagebox.showerror("Error", "No similarities computed.")
            return
        
        top = tk.Toplevel(self.root)
        top.title("Counterfactual Tumors")
        
        canvas = tk.Canvas(top)
        scrollbar = tk.Scrollbar(top, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        
        inner_frame = tk.Frame(canvas)
        canvas.create_window((0, 0), window=inner_frame, anchor="nw")
        
        for i, idx in enumerate(self.valid_indices):
            img_frame = tk.Frame(inner_frame)
            img_frame.pack(pady=10)
            
            tk.Label(img_frame, text=f"Image {idx+1}").pack()
            
            img = cv2.imread(self.image_paths[idx], cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
            img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            
            orig_feat = np.array(self.tumor_features[idx])
            similarities = self.similarity_matrix[i].copy()
            similarities[i] = -1
            j = np.argmax(similarities)
            match_feat = np.array(self.tumor_features[self.valid_indices[j]])
            
            alpha = 0.5
            cf_feat = alpha * orig_feat + (1 - alpha) * match_feat
            
            x1, y1, x2, y2 = [int(v * 512) for v in [
                orig_feat[0] - orig_feat[2]/2, orig_feat[1] - orig_feat[3]/2,
                orig_feat[0] + orig_feat[2]/2, orig_feat[1] + orig_feat[3]/2
            ]]
            cv2.rectangle(img_color, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            cf_x1, cf_y1, cf_x2, cf_y2 = [int(v * 512) for v in [
                cf_feat[0] - cf_feat[2]/2, cf_feat[1] - cf_feat[3]/2,
                cf_feat[0] + cf_feat[2]/2, cf_feat[1] + cf_feat[3]/2
            ]]
            cv2.rectangle(img_color, (cf_x1, cf_y1), (cf_x2, cf_y2), (0, 0, 255), 2)
            
            img_pil = Image.fromarray(img_color)
            img_pil.thumbnail((300, 225))
            img_tk = ImageTk.PhotoImage(img_pil)
            
            label = tk.Label(img_frame, image=img_tk)
            label.image = img_tk
            label.pack()
        
        inner_frame.update_idletasks()
        canvas.config(scrollregion=canvas.bbox("all"))
        
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind("<MouseWheel>", _on_mousewheel)

    def show_emotional_response(self):
        """Update stress levels for detected tumors."""
        if not any(self.tumor_features):
            messagebox.showerror("Error", "No tumors detected.")
            return
        for idx, feature in enumerate(self.tumor_features):
            if feature:
                stress = self.compute_stress(feature)
                self.stress_labels[idx].configure(text=f"Stress: {stress:.1f}%")

if __name__ == "__main__":
    root = tk.Tk()
    app = TumorDetectorApp(root)
    root.mainloop()