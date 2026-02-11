import numpy as np
import pandas as pd
import os
import nibabel as nib
import tkinter as tk
from tkinter import filedialog
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score
import matplotlib.pyplot as plt
from scipy.ndimage import label as scipy_label, gaussian_filter, uniform_filter
import logging

# Configure matplotlib font
plt.rcParams["font.family"] = ["Arial", "Helvetica", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("Pancreatic_CT_ITH_Analysis.log", encoding="utf-8"), 
              logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


class PancreaticCT_ITHAnalyzer:
    def __init__(
        self, 
        hu_min=-100,  
        hu_max=200,   
        smooth_sigma=1.2,  
        max_cluster_ratio=0.85,  
        connectivity=6,
        min_ith=0.01  # New: Minimum ITH value parameter
    ):
        self.image_dir = None
        self.mask_dir = None
        self.output_dir = None
        self.results = []
        self.hu_min = hu_min
        self.hu_max = hu_max
        self.smooth_sigma = smooth_sigma
        self.max_cluster_ratio = max_cluster_ratio
        self.min_ith = min_ith  # Store minimum ITH value
        
        # 6-connectivity structure
        self.connect_struct = np.array([
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
        ])

    def select_folders(self):
        """Select folders via GUI dialog"""
        root = tk.Tk()
        root.withdraw()
        self.image_dir = filedialog.askdirectory(title="Select Pancreatic Cancer CT Image Folder (.nii.gz)")
        self.mask_dir = filedialog.askdirectory(title="Select Tumor ROI Folder (.nii.gz)")
        self.output_dir = filedialog.askdirectory(title="Select Result Output Folder")
        
        if not all([self.image_dir, self.mask_dir, self.output_dir]):
            logger.error("Incomplete folder selection, program exits")
            print("❌ Incomplete folder selection, program exits")
            return False
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"Folders selected:\nCT Image Directory: {self.image_dir}\nROI Directory: {self.mask_dir}\nOutput Directory: {self.output_dir}")
        return True

    def find_best_cluster_number(self, X, cluster_range):
        """Determine optimal number of clusters"""
        ch_scores = []
        for n in cluster_range:
            kmeans = KMeans(n_clusters=n, random_state=42, n_init=5)
            labels = kmeans.fit_predict(X)
            ch_score = calinski_harabasz_score(X, labels)
            ch_scores.append(ch_score)
            logger.debug(f"Number of clusters={n}, Calinski-Harabasz score={ch_score:.2f}")

        if not ch_scores:
            logger.warning("No valid CH scores calculated, using default cluster number=2")
            return 2, ch_scores, list(cluster_range)
        
        max_score_idx = np.argmax(ch_scores)
        best_num = cluster_range[max_score_idx]
        logger.info(f"Optimal number of clusters={best_num} (Highest CH score: {ch_scores[max_score_idx]:.2f})")
        return best_num, ch_scores, list(cluster_range)

    def calculate_tumor_volume(self, mask_data, voxel_volume):
        """Calculate tumor volume"""
        tumor_voxels = np.sum(mask_data > 0)
        return tumor_voxels * voxel_volume

    def enhance_features(self, image_data, tumor_mask, voxel_dims):
        """Enhance features"""
        window_size = np.round(2 / np.array(voxel_dims)).astype(int)
        window_size = np.maximum(window_size, 1)
        window_size = np.minimum(window_size, 5)
        
        local_mean = uniform_filter(image_data, size=window_size)
        local_var = uniform_filter(image_data**2, size=window_size) - local_mean**2
        local_std = np.sqrt(np.maximum(local_var, 0))
        
        tumor_local_std = local_std[tumor_mask].reshape(-1, 1)
        tumor_local_std = (tumor_local_std - np.mean(tumor_local_std)) / (np.std(tumor_local_std) + 1e-6)
        return tumor_local_std

    def calculate_ith_score(self, cluster_map, mask_data, voxel_volume, patient_id):
        """Calculate ITH score (Ensure minimum value is 0.01)"""
        tumor_mask = mask_data > 0
        total_volume = self.calculate_tumor_volume(mask_data, voxel_volume)
        
        # Return minimum value 0.01 when tumor volume is 0
        if total_volume == 0:
            logger.warning(f"Patient {patient_id}: Tumor volume=0, ITH={self.min_ith} (Invalid sample)")
            return self.min_ith, 1.0

        unique_labels = np.unique(cluster_map[tumor_mask])
        unique_labels = unique_labels[unique_labels != 0]
        if len(unique_labels) < 2:
            logger.warning(f"Patient {patient_id}: Insufficient valid clusters (<2), ITH set to minimum value {self.min_ith}")
            return self.min_ith, 1.0  # Return minimum value when cluster number is insufficient

        sum_vi_max_over_ni = 0.0  
        cluster_volumes = []      

        for label in unique_labels:
            cluster_mask = (cluster_map == label) & tumor_mask
            labeled_regions, num_regions = scipy_label(cluster_mask, structure=self.connect_struct)
            
            cluster_total_vol = np.sum(cluster_mask) * voxel_volume
            cluster_volumes.append(cluster_total_vol)
            
            if num_regions == 0:
                continue  
            
            region_volumes = []
            for i in range(1, num_regions + 1):
                region_vol = np.sum(labeled_regions == i) * voxel_volume
                region_volumes.append(region_vol)
            
            if not region_volumes:
                continue  
            
            max_region_vol = max(region_volumes)  
            sum_vi_max_over_ni += max_region_vol / num_regions  

        # Core formula to calculate ITH
        ith_score = 1 - (sum_vi_max_over_ni / total_volume)
        # Key modification: Force ITH to be no less than minimum value 0.01
        ith_score = np.clip(ith_score, self.min_ith, 1.0)

        # Calculate maximum cluster ratio
        max_cluster_ratio = 0.0
        if cluster_volumes:
            max_cluster_vol = max(cluster_volumes)
            max_cluster_ratio = max_cluster_vol / total_volume

        logger.info(f"Patient {patient_id}: ITH calculation completed | ITH={ith_score:.6f} | Max cluster ratio={max_cluster_ratio:.2%}")
        return ith_score, max_cluster_ratio

    def plot_cluster_analysis(self, patient_id, cluster_range, ch_scores, best_num, cluster_volumes, save_dir):
        """Generate visualization results"""
        # 1. Calinski-Harabasz score curve
        plt.figure(figsize=(8, 5))
        plt.plot(cluster_range, ch_scores, 'bo-', markersize=8, label='Calinski-Harabasz Score')
        plt.axvline(x=best_num, color='r', linestyle='--', label=f'Optimal Clusters: {best_num}')
        plt.xlabel('Number of Clusters', fontsize=12)
        plt.ylabel('Calinski-Harabasz Score', fontsize=12)
        plt.title(f'Pancreatic Cancer Patient {patient_id} - CH Score Curve', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(save_dir, f'{patient_id}_ch_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Cluster volume distribution pie chart
        if len(cluster_volumes) > 0:
            total_vol = sum(cluster_volumes)
            labels = [f'Cluster {i+1}\n{vol/total_vol*100:.1f}%' 
                      for i, vol in enumerate(cluster_volumes)]
            
            plt.figure(figsize=(6, 6))
            plt.pie(
                cluster_volumes, 
                labels=labels, 
                autopct='%1.1f%%',
                startangle=90, 
                colors=plt.cm.Set3(np.linspace(0, 1, len(cluster_volumes)))
            )
            plt.title(f'Patient {patient_id} - Cluster Volume Distribution', fontsize=14)
            plt.savefig(os.path.join(save_dir, f'{patient_id}_cluster_pie.png'), dpi=300, bbox_inches='tight')
            plt.close()

    def process_single_patient(self, image_file):
        """Process single patient data"""
        patient_id = os.path.splitext(os.path.splitext(image_file)[0])[0]
        logger.info(f"\n===== Processing Pancreatic Cancer Patient: {patient_id} (Venous Phase CT) =====")
        print(f"\n===== Processing Patient: {patient_id} (Venous Phase CT) =====")

        image_path = os.path.join(self.image_dir, image_file)
        mask_path = os.path.join(self.mask_dir, image_file)
        patient_out_dir = os.path.join(self.output_dir, patient_id)
        os.makedirs(patient_out_dir, exist_ok=True)

        try:
            image_nii = nib.load(image_path)
            mask_nii = nib.load(mask_path)
            image_data = image_nii.get_fdata().astype(np.float32)
            mask_data = mask_nii.get_fdata().astype(np.uint8)

            if image_data.shape != mask_data.shape:
                logger.error(f"Patient {patient_id}: Mismatched CT and ROI dimensions, skipped")
                print(f"❌ Patient {patient_id}: Mismatched CT and ROI dimensions, skipped")
                return

            image_data = image_data * (mask_data > 0)
            image_data = np.clip(image_data, self.hu_min, self.hu_max)
            if self.smooth_sigma > 0:
                image_data = gaussian_filter(image_data, sigma=self.smooth_sigma)
                logger.debug(f"Patient {patient_id}: Applied Gaussian smoothing (sigma={self.smooth_sigma})")

            tumor_mask = mask_data > 0
            tumor_voxels = np.sum(tumor_mask)
            if tumor_voxels < 150:
                logger.warning(f"Patient {patient_id}: Tumor voxels={tumor_voxels}<150, skipped")
                print(f"⚠️ Patient {patient_id}: Tumor voxels<150, skipped")
                return

            # Dynamically set cluster range
            if tumor_voxels < 1000:
                current_cluster_range = range(2, 7)
                logger.info(f"Patient {patient_id}: Tumor voxels={tumor_voxels}<1000, using cluster range 2-6")
                print(f"Tumor voxels={tumor_voxels}<1000, using cluster range 2-6")
            else:
                current_cluster_range = range(2, 10)
                logger.info(f"Patient {patient_id}: Tumor voxels={tumor_voxels}≥1000, using cluster range 2-9")
                print(f"Tumor voxels={tumor_voxels}≥1000, using cluster range 2-9")

            voxel_dims = image_nii.header.get_zooms()
            voxel_volume = np.prod(voxel_dims)
            tumor_volume = tumor_voxels * voxel_volume
            logger.info(f"Patient {patient_id}: Voxel dimensions={voxel_dims}mm | Tumor volume={tumor_volume:.2f}mm³")
            print(f"Voxel dimensions: {voxel_dims}mm, Tumor volume: {tumor_volume:.2f}mm³")

            tumor_indices = np.argwhere(tumor_mask)
            x = tumor_indices[:, 0] / mask_data.shape[0]
            y = tumor_indices[:, 1] / mask_data.shape[1]
            z = tumor_indices[:, 2] / mask_data.shape[2]
            coords = np.column_stack((x, y, z))

            tumor_pixels = image_data[tumor_mask].reshape(-1, 1)
            tumor_pixels = (tumor_pixels - np.mean(tumor_pixels)) / (np.std(tumor_pixels) + 1e-6)

            tumor_local_std = self.enhance_features(image_data, tumor_mask, voxel_dims)

            tumor_features = np.hstack((tumor_pixels, coords, tumor_local_std))
            logger.debug(f"Patient {patient_id}: Feature dimensions={tumor_features.shape}")

            best_cluster_num, ch_scores, cluster_range = self.find_best_cluster_number(tumor_features, current_cluster_range)
            print(f"Optimal number of clusters: {best_cluster_num} (Based on CH score)")

            kmeans = KMeans(n_clusters=best_cluster_num, random_state=42, n_init=5)
            clusters = kmeans.fit_predict(tumor_features)

            cluster_map = np.zeros_like(mask_data, dtype=np.int16)
            cluster_map[tumor_mask] = clusters + 1
            cluster_nii = nib.Nifti1Image(cluster_map, image_nii.affine, image_nii.header)
            cluster_path = os.path.join(patient_out_dir, f'{patient_id}_pancreatic_cluster.nii.gz')
            nib.save(cluster_nii, cluster_path)
            logger.info(f"Patient {patient_id}: Cluster results saved to {cluster_path}")
            print(f"✅ 3D cluster results saved: {cluster_path}")

            ith_score, max_cluster_ratio = self.calculate_ith_score(cluster_map, mask_data, voxel_volume, patient_id)

            unique_labels = np.unique(cluster_map[tumor_mask])
            unique_labels = unique_labels[unique_labels != 0]
            cluster_volumes = [np.sum((cluster_map == label) & tumor_mask) * voxel_volume for label in unique_labels]

            self.plot_cluster_analysis(patient_id, cluster_range, ch_scores, best_cluster_num, cluster_volumes, patient_out_dir)

            cluster_quality = "Pass" if max_cluster_ratio <= self.max_cluster_ratio else "Warning (Excessively high single cluster ratio)"
            self.results.append({
                'Patient ID': patient_id,
                'Tumor Voxels': tumor_voxels,
                'Optimal Cluster Number': best_cluster_num,
                'ITH Score (Venous Phase)': round(ith_score, 4),
                'Tumor Volume (mm³)': round(tumor_volume, 2),
                'Max Cluster Ratio': round(max_cluster_ratio, 4),
                'Cluster Quality': cluster_quality
            })
            # Display ITH with 4 decimal places (minimum value 0.01 guaranteed)
            print(f"✅ Processing completed: ITH={ith_score:.4f}, Max cluster ratio={max_cluster_ratio:.2%}, Quality={cluster_quality}")

        except Exception as e:
            logger.error(f"Patient {patient_id}: Processing failed - {str(e)}", exc_info=True)
            print(f"❌ Patient {patient_id}: Processing failed - {str(e)}")

    def run_batch_analysis(self):
        """Batch process multiple patients"""
        image_files = [f for f in os.listdir(self.image_dir) if f.endswith('.nii.gz')]
        if not image_files:
            logger.error("No pancreatic cancer CT files in .nii.gz format found")
            print("❌ No pancreatic cancer CT files found")
            return

        logger.info(f"Found {len(image_files)} pancreatic cancer venous phase CT files, starting batch analysis")
        print(f"\nFound {len(image_files)} pancreatic cancer patients (Venous Phase CT), starting analysis...")
        for img_file in image_files:
            self.process_single_patient(img_file)

        if self.results:
            df = pd.DataFrame(self.results)[[
                'Patient ID', 'Tumor Voxels',
                'Optimal Cluster Number',
                'ITH Score (Venous Phase)', 
                'Tumor Volume (mm³)', 
                'Max Cluster Ratio', 
                'Cluster Quality'
            ]]
            
            output_path = os.path.join(self.output_dir, 'Pancreatic_Cancer_Venous_CT_ITH_Analysis_Results.xlsx')
            df.to_excel(output_path, index=False)
            logger.info(f"Batch analysis completed, results saved to {output_path}")
            print(f"\n📊 Results saved: {output_path}")
            
            # Statistical summary for valid ITH (based on minimum value 0.01)
            valid_ith = df[df['ITH Score (Venous Phase)'] >= self.min_ith]['ITH Score (Venous Phase)']
            print(f"Statistical Summary:\n- Valid ITH results: {len(valid_ith)}/{len(df)}\n- Mean ITH: {valid_ith.mean():.4f}\n- ITH Standard Deviation: {valid_ith.std():.4f}\n- Passed cluster ratio: {len(df[df['Cluster Quality']=='Pass'])/len(df)*100:.1f}%")
        else:
            logger.warning("No valid ITH analysis results")
            print("\n⚠️ No valid results")


if __name__ == "__main__":
    # Initialize with minimum ITH value set to 0.01 explicitly
    analyzer = PancreaticCT_ITHAnalyzer(
        hu_min=-100,
        hu_max=200,
        smooth_sigma=1.2,
        max_cluster_ratio=0.85,
        connectivity=6,
        min_ith=0.01  # Explicitly set minimum ITH value
    )
    
    if analyzer.select_folders():
        analyzer.run_batch_analysis()
    print("\nPancreatic Cancer Venous Phase CT ITH Analysis Completed")