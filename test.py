import os
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from sklearn.metrics import mutual_info_score
from skimage import measure, feature
import warnings

warnings.filterwarnings('ignore')

# Set matplotlib to use English
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class FacePrivacyAnalyzer:
    def __init__(self, original_path, processed_path):
        self.original_path = original_path
        self.processed_path = processed_path
        self.results = []

    def calculate_shannon_entropy(self, image):
        """Calculate Shannon entropy"""
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        hist, _ = np.histogram(image, bins=256, range=(0, 256))
        hist = hist / hist.sum()
        hist = hist[hist > 0]
        entropy = -np.sum(hist * np.log2(hist))
        return entropy

    def extract_global_features(self, image):
        """Extract global features for authenticity analysis"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        features = {}

        # 1. Global statistical features
        features['mean_intensity'] = np.mean(gray)
        features['std_intensity'] = np.std(gray)
        features['skewness'] = stats.skew(gray.flatten())
        features['kurtosis'] = stats.kurtosis(gray.flatten())

        # 2. Texture features (GLCM)
        try:
            glcm = feature.graycomatrix(gray, [5], [0], 256, symmetric=True, normed=True)
            features['contrast'] = feature.graycoprops(glcm, 'contrast')[0, 0]
            features['dissimilarity'] = feature.graycoprops(glcm, 'dissimilarity')[0, 0]
            features['homogeneity'] = feature.graycoprops(glcm, 'homogeneity')[0, 0]
            features['energy'] = feature.graycoprops(glcm, 'energy')[0, 0]
        except:
            features['contrast'] = 0
            features['dissimilarity'] = 0
            features['homogeneity'] = 0
            features['energy'] = 0

        # 3. Frequency domain features
        fft = np.fft.fft2(gray)
        fft_shift = np.fft.fftshift(fft)
        magnitude_spectrum = np.log(np.abs(fft_shift) + 1)

        features['freq_mean'] = np.mean(magnitude_spectrum)
        features['freq_std'] = np.std(magnitude_spectrum)

        # 4. Gradient features
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)

        features['gradient_mean'] = np.mean(gradient_magnitude)
        features['gradient_std'] = np.std(gradient_magnitude)

        return features

    def calculate_global_similarity(self, orig_features, proc_features):
        """Calculate global feature similarity"""
        similarities = {}

        # Statistical feature similarity
        stat_features = ['mean_intensity', 'std_intensity', 'skewness', 'kurtosis']
        stat_similarities = []
        for feature in stat_features:
            orig_val = orig_features[feature]
            proc_val = proc_features[feature]
            if orig_val != 0:
                similarity = 1 - abs(orig_val - proc_val) / abs(orig_val)
            else:
                similarity = 1 if proc_val == 0 else 0
            stat_similarities.append(max(0, similarity))
        similarities['statistical'] = np.mean(stat_similarities)

        # Texture feature similarity
        texture_features = ['contrast', 'dissimilarity', 'homogeneity', 'energy']
        texture_similarities = []
        for feature in texture_features:
            orig_val = orig_features[feature]
            proc_val = proc_features[feature]
            if orig_val != 0:
                similarity = 1 - abs(orig_val - proc_val) / abs(orig_val)
            else:
                similarity = 1 if proc_val == 0 else 0
            texture_similarities.append(max(0, similarity))
        similarities['texture'] = np.mean(texture_similarities)

        # Frequency domain feature similarity
        freq_features = ['freq_mean', 'freq_std']
        freq_similarities = []
        for feature in freq_features:
            orig_val = orig_features[feature]
            proc_val = proc_features[feature]
            if orig_val != 0:
                similarity = 1 - abs(orig_val - proc_val) / abs(orig_val)
            else:
                similarity = 1 if proc_val == 0 else 0
            freq_similarities.append(max(0, similarity))
        similarities['frequency'] = np.mean(freq_similarities)

        # Gradient feature similarity
        grad_features = ['gradient_mean', 'gradient_std']
        grad_similarities = []
        for feature in grad_features:
            orig_val = orig_features[feature]
            proc_val = proc_features[feature]
            if orig_val != 0:
                similarity = 1 - abs(orig_val - proc_val) / abs(orig_val)
            else:
                similarity = 1 if proc_val == 0 else 0
            grad_similarities.append(max(0, similarity))
        similarities['gradient'] = np.mean(grad_similarities)

        return similarities

    def calculate_structural_similarity(self, original, processed):
        """Calculate structural similarity (not dependent on pixel details)"""
        if len(original.shape) == 3:
            orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        else:
            orig_gray = original.copy()

        if len(processed.shape) == 3:
            proc_gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        else:
            proc_gray = processed.copy()

        # 1. Low resolution structural similarity
        orig_low = cv2.resize(orig_gray, (32, 32))
        proc_low = cv2.resize(proc_gray, (32, 32))

        # Calculate structural similarity
        mean_orig = np.mean(orig_low)
        mean_proc = np.mean(proc_low)

        var_orig = np.var(orig_low)
        var_proc = np.var(proc_low)

        cov = np.mean((orig_low - mean_orig) * (proc_low - mean_proc))

        c1 = 0.01 ** 2
        c2 = 0.03 ** 2

        ssim = ((2 * mean_orig * mean_proc + c1) * (2 * cov + c2)) / \
               ((mean_orig ** 2 + mean_proc ** 2 + c1) * (var_orig + var_proc + c2))

        return max(0, ssim)

    def calculate_face_region_similarity(self, original, processed):
        """Calculate face region similarity"""
        # Use Haar cascade for face detection
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        if len(original.shape) == 3:
            orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        else:
            orig_gray = original.copy()

        if len(processed.shape) == 3:
            proc_gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        else:
            proc_gray = processed.copy()

        # Detect faces
        orig_faces = face_cascade.detectMultiScale(orig_gray, 1.3, 5)
        proc_faces = face_cascade.detectMultiScale(proc_gray, 1.3, 5)

        if len(orig_faces) == 0 or len(proc_faces) == 0:
            # If no faces detected, use the entire image
            return self.calculate_structural_similarity(original, processed)

        # Use the largest face region
        orig_face = max(orig_faces, key=lambda x: x[2] * x[3])
        proc_face = max(proc_faces, key=lambda x: x[2] * x[3])

        # Extract face regions
        x1, y1, w1, h1 = orig_face
        x2, y2, w2, h2 = proc_face

        orig_face_region = orig_gray[y1:y1 + h1, x1:x1 + w1]
        proc_face_region = proc_gray[y2:y2 + h2, x2:x2 + w2]

        # Resize to maintain consistency
        size = (64, 64)
        orig_face_resized = cv2.resize(orig_face_region, size)
        proc_face_resized = cv2.resize(proc_face_region, size)

        # Calculate face region similarity
        return self.calculate_structural_similarity(orig_face_resized, proc_face_resized)

    def calculate_recognizability_score(self, original, processed):
        """Calculate recognizability score (comprehensive authenticity metrics)"""
        # 1. Global feature similarity (40%)
        orig_features = self.extract_global_features(original)
        proc_features = self.extract_global_features(processed)
        global_sim = self.calculate_global_similarity(orig_features, proc_features)

        global_score = (global_sim['statistical'] * 0.3 +
                        global_sim['texture'] * 0.3 +
                        global_sim['frequency'] * 0.2 +
                        global_sim['gradient'] * 0.2)

        # 2. Structural similarity (30%)
        struct_score = self.calculate_structural_similarity(original, processed)

        # 3. Face region similarity (30%)
        face_score = self.calculate_face_region_similarity(original, processed)

        # Comprehensive score
        recognizability = global_score * 0.4 + struct_score * 0.3 + face_score * 0.3

        return {
            'global_similarity': global_score,
            'structural_similarity': struct_score,
            'face_similarity': face_score,
            'recognizability_score': recognizability,
            'detailed_global': global_sim
        }

    def calculate_local_entropy(self, image, window_size=15):
        """Calculate local entropy"""
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        entropy_map = np.zeros_like(image, dtype=np.float32)

        for i in range(window_size // 2, image.shape[0] - window_size // 2):
            for j in range(window_size // 2, image.shape[1] - window_size // 2):
                window = image[i - window_size // 2:i + window_size // 2 + 1,
                         j - window_size // 2:j + window_size // 2 + 1]
                entropy_map[i, j] = measure.shannon_entropy(window)

        return entropy_map

    def detect_entropy_anomalies(self, entropy_map, threshold_percentile=95):
        """Detect entropy anomaly regions"""
        threshold = np.percentile(entropy_map, threshold_percentile)
        anomaly_mask = entropy_map > threshold
        anomaly_ratio = np.sum(anomaly_mask) / entropy_map.size
        return anomaly_ratio, threshold

    def analyze_single_pair(self, image_name):
        """Analyze single image pair"""
        orig_path = os.path.join(self.original_path, image_name)
        proc_path = os.path.join(self.processed_path, image_name)

        if not os.path.exists(orig_path) or not os.path.exists(proc_path):
            print(f"Warning: Cannot find image pair {image_name}")
            return None

        original = cv2.imread(orig_path)
        processed = cv2.imread(proc_path)

        if original is None or processed is None:
            print(f"Warning: Cannot read image pair {image_name}")
            return None

        try:
            # Naturalness analysis
            orig_entropy = self.calculate_shannon_entropy(original)
            proc_entropy = self.calculate_shannon_entropy(processed)
            entropy_diff = abs(orig_entropy - proc_entropy)

            local_entropy_map = self.calculate_local_entropy(processed)
            anomaly_ratio, _ = self.detect_entropy_anomalies(local_entropy_map)

            # Authenticity analysis (recognizability)
            recognizability_metrics = self.calculate_recognizability_score(original, processed)

            # Naturalness score
            naturalness_score = max(0, 1 - (entropy_diff / 2) - anomaly_ratio)

            result = {
                'image_name': image_name,
                'original_entropy': orig_entropy,
                'processed_entropy': proc_entropy,
                'entropy_difference': entropy_diff,
                'anomaly_ratio': anomaly_ratio,
                'naturalness_score': naturalness_score,

                # Authenticity related metrics
                'global_similarity': recognizability_metrics['global_similarity'],
                'structural_similarity': recognizability_metrics['structural_similarity'],
                'face_similarity': recognizability_metrics['face_similarity'],
                'recognizability_score': recognizability_metrics['recognizability_score'],

                # Detailed global feature similarity
                'statistical_similarity': recognizability_metrics['detailed_global']['statistical'],
                'texture_similarity': recognizability_metrics['detailed_global']['texture'],
                'frequency_similarity': recognizability_metrics['detailed_global']['frequency'],
                'gradient_similarity': recognizability_metrics['detailed_global']['gradient']
            }

            return result

        except Exception as e:
            print(f"Error processing image {image_name}: {str(e)}")
            return None

    def analyze_all_images(self):
        """Analyze all image pairs"""
        print("Starting image analysis...")

        original_images = set()
        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
            original_images.update([f for f in os.listdir(self.original_path)
                                    if f.lower().endswith(ext)])

        processed_images = set()
        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
            processed_images.update([f for f in os.listdir(self.processed_path)
                                     if f.lower().endswith(ext)])

        common_images = original_images.intersection(processed_images)
        print(f"Found {len(common_images)} matching image pairs")

        self.results = []
        for i, image_name in enumerate(sorted(common_images)):
            print(f"Processing: {i + 1}/{len(common_images)} - {image_name}")
            result = self.analyze_single_pair(image_name)
            if result:
                self.results.append(result)

        print(f"Successfully analyzed {len(self.results)} image pairs")

    def generate_report(self):
        """Generate analysis report"""
        if not self.results:
            print("No analysis results available for report generation")
            return

        df = pd.DataFrame(self.results)

        print("=" * 70)
        print("Face Privacy Protection Effectiveness Analysis Report")
        print("=" * 70)

        print(f"\nTotal analyzed images: {len(self.results)}")

        # Naturalness analysis
        print("\n" + "=" * 50)
        print("1. Naturalness Analysis (Whether images look natural)")
        print("=" * 50)
        print(f"Average entropy difference: {df['entropy_difference'].mean():.4f} ± {df['entropy_difference'].std():.4f}")
        print(f"Average anomaly region ratio: {df['anomaly_ratio'].mean() * 100:.2f}%")
        print(f"Average naturalness score: {df['naturalness_score'].mean():.4f}")

        natural_good = (df['naturalness_score'] > 0.8).sum()
        natural_fair = ((df['naturalness_score'] > 0.6) & (df['naturalness_score'] <= 0.8)).sum()
        natural_poor = (df['naturalness_score'] <= 0.6).sum()

        print(f"\nNaturalness rating distribution:")
        print(f"  Excellent (>0.8): {natural_good}/{len(self.results)} ({natural_good / len(self.results) * 100:.1f}%)")
        print(f"  Good (0.6-0.8): {natural_fair}/{len(self.results)} ({natural_fair / len(self.results) * 100:.1f}%)")
        print(f"  Needs improvement (≤0.6): {natural_poor}/{len(self.results)} ({natural_poor / len(self.results) * 100:.1f}%)")

        # Authenticity analysis (recognizability)
        print("\n" + "=" * 50)
        print("2. Authenticity Analysis (Whether acquaintances can recognize)")
        print("=" * 50)
        print(f"Average recognizability score: {df['recognizability_score'].mean():.4f} ± {df['recognizability_score'].std():.4f}")
        print(f"Average global feature similarity: {df['global_similarity'].mean():.4f}")
        print(f"Average structural similarity: {df['structural_similarity'].mean():.4f}")
        print(f"Average face region similarity: {df['face_similarity'].mean():.4f}")

        print(f"\nDetailed feature similarity:")
        print(f"  Statistical feature similarity: {df['statistical_similarity'].mean():.4f}")
        print(f"  Texture feature similarity: {df['texture_similarity'].mean():.4f}")
        print(f"  Frequency feature similarity: {df['frequency_similarity'].mean():.4f}")
        print(f"  Gradient feature similarity: {df['gradient_similarity'].mean():.4f}")

        recogn_excellent = (df['recognizability_score'] > 0.8).sum()
        recogn_good = ((df['recognizability_score'] > 0.6) & (df['recognizability_score'] <= 0.8)).sum()
        recogn_fair = ((df['recognizability_score'] > 0.4) & (df['recognizability_score'] <= 0.6)).sum()
        recogn_poor = (df['recognizability_score'] <= 0.4).sum()

        print(f"\nRecognizability rating distribution:")
        print(
            f"  Excellent (>0.8): {recogn_excellent}/{len(self.results)} ({recogn_excellent / len(self.results) * 100:.1f}%)")
        print(f"  Good (0.6-0.8): {recogn_good}/{len(self.results)} ({recogn_good / len(self.results) * 100:.1f}%)")
        print(f"  Fair (0.4-0.6): {recogn_fair}/{len(self.results)} ({recogn_fair / len(self.results) * 100:.1f}%)")
        print(f"  Poor (≤0.4): {recogn_poor}/{len(self.results)} ({recogn_poor / len(self.results) * 100:.1f}%)")

        # Comprehensive evaluation
        print("\n" + "=" * 50)
        print("3. Comprehensive Evaluation")
        print("=" * 50)

        overall_naturalness = df['naturalness_score'].mean()
        overall_recognizability = df['recognizability_score'].mean()

        print(f"Overall naturalness score: {overall_naturalness:.4f}")
        print(f"Overall recognizability score: {overall_recognizability:.4f}")

        # Rating logic
        if overall_naturalness > 0.8:
            nat_grade = "Excellent ✓"
        elif overall_naturalness > 0.6:
            nat_grade = "Good ○"
        else:
            nat_grade = "Needs improvement ✗"

        if overall_recognizability > 0.7:
            rec_grade = "Excellent ✓ (acquaintances can easily recognize)"
        elif overall_recognizability > 0.5:
            rec_grade = "Good ○ (acquaintances can recognize)"
        elif overall_recognizability > 0.3:
            rec_grade = "Fair △ (acquaintances might recognize)"
        else:
            rec_grade = "Poor ✗ (acquaintances have difficulty recognizing)"

        print(f"\nFinal rating:")
        print(f"  Naturalness: {nat_grade}")
        print(f"  Recognizability: {rec_grade}")

        # Recommendations
        print(f"\nRecommendations:")
        if overall_naturalness < 0.6:
            print("  - Image naturalness needs improvement, obvious processing artifacts may exist")
        if overall_recognizability < 0.4:
            print("  - Low recognizability, even acquaintances may have difficulty recognizing")
        elif overall_recognizability > 0.8:
            print("  - Good recognizability, acquaintances should be able to easily recognize")

        return df

    def visualize_results(self, df=None):
        """Visualize analysis results"""
        if df is None:
            if not self.results:
                print("No results available for visualization")
                return
            df = pd.DataFrame(self.results)

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Face Privacy Protection Effectiveness Analysis', fontsize=16, fontweight='bold')

        # Naturalness score distribution
        axes[0, 0].hist(df['naturalness_score'], bins=20, alpha=0.7, edgecolor='black', color='skyblue')
        axes[0, 0].axvline(df['naturalness_score'].mean(), color='red', linestyle='--',
                           label=f'Mean: {df["naturalness_score"].mean():.3f}')
        axes[0, 0].set_xlabel('Naturalness Score')
        axes[0, 0].set_ylabel('Number of Images')
        axes[0, 0].set_title('Naturalness Score Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Recognizability score distribution
        axes[0, 1].hist(df['recognizability_score'], bins=20, alpha=0.7, edgecolor='black', color='lightgreen')
        axes[0, 1].axvline(df['recognizability_score'].mean(), color='red', linestyle='--',
                           label=f'Mean: {df["recognizability_score"].mean():.3f}')
        axes[0, 1].set_xlabel('Recognizability Score')
        axes[0, 1].set_ylabel('Number of Images')
        axes[0, 1].set_title('Recognizability Score Distribution')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Naturalness vs Recognizability scatter plot
        axes[0, 2].scatter(df['naturalness_score'], df['recognizability_score'],
                           alpha=0.6, s=50, color='purple')
        axes[0, 2].set_xlabel('Naturalness Score')
        axes[0, 2].set_ylabel('Recognizability Score')
        axes[0, 2].set_title('Naturalness vs Recognizability Relationship')
        axes[0, 2].grid(True, alpha=0.3)

        # Different similarity metrics comparison
        similarity_metrics = ['global_similarity', 'structural_similarity', 'face_similarity']
        similarity_values = [df[metric].mean() for metric in similarity_metrics]
        similarity_labels = ['Global Features', 'Structural Features', 'Face Region']

        bars = axes[1, 0].bar(similarity_labels, similarity_values,
                              color=['orange', 'green', 'blue'], alpha=0.7)
        axes[1, 0].set_ylabel('Average Similarity')
        axes[1, 0].set_title('Different Feature Similarity Comparison')
        axes[1, 0].set_ylim(0, 1)

        # Add value labels on bars
        for bar, value in zip(bars, similarity_values):
            axes[1, 0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                            f'{value:.3f}', ha='center', va='bottom')

        # Detailed feature similarity radar chart effect
        feature_names = ['Statistical', 'Texture', 'Frequency', 'Gradient']
        feature_values = [df['statistical_similarity'].mean(),
                          df['texture_similarity'].mean(),
                          df['frequency_similarity'].mean(),
                          df['gradient_similarity'].mean()]

        bars2 = axes[1, 1].bar(range(len(feature_names)), feature_values,
                               color=['red', 'yellow', 'cyan', 'magenta'], alpha=0.7)
        axes[1, 1].set_xticks(range(len(feature_names)))
        axes[1, 1].set_xticklabels(feature_names, rotation=45)
        axes[1, 1].set_ylabel('Average Similarity')
        axes[1, 1].set_title('Detailed Feature Similarity Analysis')
        axes[1, 1].set_ylim(0, 1)

        # Add value labels
        for bar, value in zip(bars2, feature_values):
            axes[1, 1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                            f'{value:.3f}', ha='center', va='bottom')

        # Rating distribution pie chart
        recogn_labels = ['Excellent\n(>0.8)', 'Good\n(0.6-0.8)', 'Fair\n(0.4-0.6)', 'Poor\n(≤0.4)']
        recogn_counts = [
            (df['recognizability_score'] > 0.8).sum(),
            ((df['recognizability_score'] > 0.6) & (df['recognizability_score'] <= 0.8)).sum(),
            ((df['recognizability_score'] > 0.4) & (df['recognizability_score'] <= 0.6)).sum(),
            (df['recognizability_score'] <= 0.4).sum()
        ]

        colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
        axes[1, 2].pie(recogn_counts, labels=recogn_labels, colors=colors, autopct='%1.1f%%')
        axes[1, 2].set_title('Recognizability Rating Distribution')

        plt.tight_layout()
        plt.savefig('/root/autodl-tmp/pulse/face_privacy_analysisreal2.png', dpi=300, bbox_inches='tight')
        plt.show()

    def save_results(self, filename='/root/autodl-tmp/pulse/face_privacy_analysisreal2.csv'):
        """Save results to CSV file"""
        if not self.results:
            print("No results to save")
            return

        df = pd.DataFrame(self.results)
        df.to_csv(filename, index=False)
        print(f"Results saved to {filename}")
        return df


def main():
    original_path = "/root/autodl-tmp/pulse/datareal"
    processed_path = "/root/autodl-tmp/pulse/runsreal2"

    if not os.path.exists(original_path):
        print(f"Error: Original image path does not exist {original_path}")
        return

    if not os.path.exists(processed_path):
        print(f"Error: Processed image path does not exist {processed_path}")
        return

    analyzer = FacePrivacyAnalyzer(original_path, processed_path)
    analyzer.analyze_all_images()
    df = analyzer.generate_report()
    analyzer.visualize_results(df)
    analyzer.save_results()

    print("\nAnalysis complete! Results saved.")


if __name__ == "__main__":
    main()