import cv2
import numpy as np
import shutil
from pathlib import Path
from PIL import Image
from tqdm import tqdm


class VideoProcessor:
    @staticmethod
    def detect_scene_changes(frames, threshold=30.0):
        """
        Detect scene changes by analyzing frame-to-frame differences.

        Uses histogram comparison to identify significant visual changes between frames.

        Args:
            frames: List of (frame, timestamp) tuples
            threshold: Threshold for scene change detection (0-100).
                      Lower = more sensitive, higher = less sensitive.
                      Default 30.0 catches major scene changes.

        Returns:
            List of scene IDs corresponding to each frame
        """
        if len(frames) <= 1:
            return [0] * len(frames)

        scene_ids = [0]
        current_scene = 0

        print(f"Detecting scene changes (threshold={threshold})...")

        for i in tqdm(range(1, len(frames)), desc="Scene detection"):
            prev_frame = frames[i-1][0]
            curr_frame = frames[i][0]

            # Convert to HSV for better color comparison
            prev_hsv = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2HSV)
            curr_hsv = cv2.cvtColor(curr_frame, cv2.COLOR_RGB2HSV)

            # Calculate histograms for each channel
            hist_diff = 0
            for channel in range(3):
                prev_hist = cv2.calcHist([prev_hsv], [channel], None, [256], [0, 256])
                curr_hist = cv2.calcHist([curr_hsv], [channel], None, [256], [0, 256])

                # Normalize histograms
                cv2.normalize(prev_hist, prev_hist, 0, 1, cv2.NORM_MINMAX)
                cv2.normalize(curr_hist, curr_hist, 0, 1, cv2.NORM_MINMAX)

                # Compare using correlation method (returns 0-1, higher = more similar)
                correlation = cv2.compareHist(prev_hist, curr_hist, cv2.HISTCMP_CORREL)

                # Convert to difference score (0-100, higher = more different)
                hist_diff += (1.0 - correlation) * 100 / 3

            # If difference exceeds threshold, it's a new scene
            if hist_diff > threshold:
                current_scene += 1

            scene_ids.append(current_scene)

        num_scenes = current_scene + 1
        print(f"Detected {num_scenes} scenes across {len(frames)} frames")

        return scene_ids
    @staticmethod
    def is_solid_or_gradient(frame, std_threshold=2, edge_density_threshold=0.001, black_threshold=5, white_threshold=250):
        """
        Detect if a frame is a solid color, gradient, or black/white frame.

        Very conservative thresholds to only skip completely blank frames:
        - Black frame detection: mean brightness < 2% (~5/255) - almost pure black
        - Solid color detection: std deviation < 2 - almost no variation
        - Edge density: < 0.1% edge pixels - essentially no detail at all

        Args:
            frame: RGB frame as numpy array
            std_threshold: Max std deviation for solid color (default: 2)
            edge_density_threshold: Max ratio of edge pixels (default: 0.001 = 0.1%)
            black_threshold: Max mean brightness for black frame (default: 5/255)
            white_threshold: Min mean brightness for white frame (default: 250/255)

        Returns:
            True if frame should be skipped, False otherwise
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        mean_brightness = np.mean(gray)
        if mean_brightness < black_threshold or mean_brightness > white_threshold:
            return True

        std_dev = np.std(gray)
        if std_dev < std_threshold:
            return True

        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.count_nonzero(edges) / edges.size
        if edge_density < edge_density_threshold:
            return True

        return False

    @staticmethod
    def clear_and_prepare_directories(base_dir="./frame_output"):
        base_path = Path(base_dir)
        saved_dir = base_path / "saved"
        skipped_dir = base_path / "skipped"

        if base_path.exists():
            shutil.rmtree(base_path)

        saved_dir.mkdir(parents=True, exist_ok=True)
        skipped_dir.mkdir(parents=True, exist_ok=True)

        return saved_dir, skipped_dir

    @staticmethod
    def extract_frames(video_path, frame_interval=1.5, skip_solid_frames=True, save_frames_to_disk=True):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        frame_step = int(fps * frame_interval)
        frames = []

        print(f"Extracting frames from {video_path}")
        print(f"Video: {duration:.2f}s, {fps:.2f} fps, sampling every {frame_interval}s")
        if skip_solid_frames:
            print("Skipping solid color and gradient frames")

        saved_dir, skipped_dir = None, None
        if save_frames_to_disk:
            saved_dir, skipped_dir = VideoProcessor.clear_and_prepare_directories()
            print(f"Saving frames to {saved_dir.parent}")

        frame_count = 0
        skipped_count = 0
        saved_count = 0
        with tqdm(total=int(duration / frame_interval)) as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_count % frame_step == 0:
                    timestamp = frame_count / fps
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    if skip_solid_frames and VideoProcessor.is_solid_or_gradient(frame_rgb):
                        skipped_count += 1
                        pbar.set_postfix({"saved": saved_count, "skipped": skipped_count})

                        if save_frames_to_disk and skipped_dir:
                            img = Image.fromarray(frame_rgb)
                            img.save(skipped_dir / f"frame_{frame_count:06d}_t{timestamp:.2f}s.jpg")
                    else:
                        frames.append((frame_rgb, timestamp))
                        saved_count += 1
                        pbar.set_postfix({"saved": saved_count, "skipped": skipped_count})

                        if save_frames_to_disk and saved_dir:
                            img = Image.fromarray(frame_rgb)
                            img.save(saved_dir / f"frame_{frame_count:06d}_t{timestamp:.2f}s.jpg")

                    pbar.update(1)
                frame_count += 1

        cap.release()
        print(f"Extracted {len(frames)} frames (skipped {skipped_count} solid/gradient frames)")
        return frames

    @staticmethod
    def get_frame_at_timestamp(video_path, timestamp):
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_number = int(timestamp * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            cap.release()
            if ret:
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return None
        except:
            return None
