#!/usr/bin/env python3
import os
import sys
import argparse
import numpy as np
import torch
import ffmpeg
from tqdm import tqdm

from transnetv2_pytorch.model import TransNetV2


class TransNetV2Torch:
    def __init__(self):
        self._input_size = (27, 48, 3)
        # assumeC the weights are located in the same directory as this script
        weights_path = os.path.join(os.path.dirname(__file__),
                                    'transnetv2-pytorch-weights.pth')
        self.model = TransNetV2()
        self.model.load_state_dict(torch.load(weights_path))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.eval().to(self.device)
        self.window_buf = None

    @torch.no_grad()
    def predict_raw(self, frames_tensor):
        logits, out = self.model(frames_tensor)
        return logits.cpu().numpy(), out["many_hot"].cpu().numpy()

    def predict_frames(self, frames: np.ndarray):
        total = len(frames)
        pad_start = np.repeat(frames[0:1], 25, axis=0)
        pad_end_len = 25 + (50 - (total % 50) if total % 50 != 0 else 50)
        pad_end = np.repeat(frames[-1:], pad_end_len, axis=0)
        padded = np.concatenate((pad_start, frames, pad_end), axis=0)

        if self.window_buf is None:
            self.window_buf = np.empty((1, 100, *self._input_size),
                                       dtype=np.uint8)

        logits_list, manyhot_list = [], []
        ptr = 0
        while ptr + 100 <= len(padded):
            self.window_buf[0] = padded[ptr:ptr + 100]
            tensor_input = torch.from_numpy(self.window_buf).to(self.device)
            l, mh = self.predict_raw(tensor_input)
            logits_list.append(l[0, 25:75, 0])
            manyhot_list.append(mh[0, 25:75, 0])

            ptr += 50
            processed = min(ptr, total)
            print(f"\r[TransNetV2] Processing video frames {processed}/{total}", end="")

        print("")  # newline after bar

        logits = np.concatenate(logits_list, axis=0)[:total]
        manyhot = np.concatenate(manyhot_list, axis=0)[:total]

        # sigmoid in numpy
        s_pred = 1.0 / (1.0 + np.exp(-logits))
        a_pred = 1.0 / (1.0 + np.exp(-manyhot))
        return s_pred, a_pred

    def predict_video(self, video_path: str):
        print(f"[TransNetV2] Extracting frames from {os.path.basename(video_path)}")
        video_stream, _ = ffmpeg.input(video_path).output(
            "pipe:", format="rawvideo", pix_fmt="rgb24", s="48x27"
        ).run(capture_stdout=True, capture_stderr=True)

        frames = np.frombuffer(video_stream, np.uint8).reshape([-1, 27, 48, 3])
        return frames, *self.predict_frames(frames)

    @staticmethod
    def predictions_to_scenes(preds, threshold=0.5):
        preds = (preds > threshold).astype(np.uint8)
        scenes, t_prev, start = [], 0, 0
        for i, t in enumerate(preds):
            if t_prev == 1 and t == 0:
                start = i
            if t_prev == 0 and t == 1 and i != 0:
                scenes.append([start, i])
            t_prev = t
        if t == 0:
            scenes.append([start, i])
        if not scenes:
            return np.array([[0, len(preds) - 1]], dtype=np.int32)
        return np.array(scenes, dtype=np.int32)

    @staticmethod
    def visualize_predictions(frames: np.ndarray, predictions):
        from PIL import Image, ImageDraw

        if isinstance(predictions, np.ndarray):
            predictions = [predictions]

        ih, iw, _ = frames.shape[1:]
        width = 25
        pad_with = width - len(frames) % width if len(frames) % width != 0 else 0

        frames = np.pad(frames, [(0, pad_with), (0, 1), (0, len(predictions)), (0, 0)])
        predictions = [np.pad(x, (0, pad_with)) for x in predictions]
        height = len(frames) // width

        img = frames.reshape([height, width, ih + 1, iw + len(predictions), 3])
        img = np.concatenate(np.split(
            np.concatenate(np.split(img, height), axis=2)[0], width
        ), axis=2)[0, :-1]

        img = Image.fromarray(img)
        draw = ImageDraw.Draw(img)
        for i, pred in enumerate(zip(*predictions)):
            x, y = i % width, i // width
            x, y = x * (iw + len(predictions)) + iw, y * (ih + 1) + ih - 1
            for j, p in enumerate(pred):
                color = [0, 0, 0]
                color[(j + 1) % 3] = 255
                value = round(p * (ih - 1))
                if value != 0:
                    draw.line((x + j, y, x + j, y - value),
                              fill=tuple(color), width=1)
        return img

    @staticmethod
    def extract_keyframes_from_scenes(s_pred, threshold=0.5):
        """
        Extract keyframes from scene middle points
        
        Args:
            s_pred: Shot cut predictions
            threshold: Threshold for scene detection
            
        Returns:
            keyframe_indices: List of frame indices for keyframes (middle of each scene)
        """
        # Get scenes using the existing method
        scenes = TransNetV2Torch.predictions_to_scenes(s_pred, threshold)
        
        keyframe_indices = []
        
        # Extract middle point of each scene
        for start_frame, end_frame in scenes:
            middle_frame = (start_frame + end_frame) // 2
            keyframe_indices.append(middle_frame)
        
        return keyframe_indices

    @staticmethod
    def save_keyframes_to_folder(video_path: str, keyframe_indices, output_dir):
        """
        Extract keyframes using optimized flow: pre-process video to 720p, then extract keyframes
        
        Args:
            video_path: Path to original video
            keyframe_indices: List of frame indices to extract
            output_dir: Directory to save keyframes
            
        Note:
            - Video is pre-processed to 720p once, then keyframes are extracted from processed video
            - Much faster than scaling individual frames
            - Output format: JPG files named as frame_index (e.g., 000123.jpg)
        """
        import os
        import tempfile
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"[TransNetV2] Processing {len(keyframe_indices)} keyframes from {os.path.basename(video_path)}")
        
        # Get video info for FPS and frame validation
        try:
            probe = ffmpeg.probe(video_path)
            video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
            fps = eval(video_info['r_frame_rate'])  # Convert "25/1" to 25.0
            duration = float(video_info['duration'])
            total_frames = int(fps * duration)
            
        except Exception as e:
            print(f"Error getting video info: {e}")
            fps = 25.0  # Default FPS
            total_frames = None
        
        # Validate keyframe indices
        valid_keyframes = []
        for frame_idx in keyframe_indices:
            if total_frames and frame_idx >= total_frames:
                print(f"[Warning] Frame {frame_idx} exceeds video length ({total_frames} frames), skipping")
                continue
            valid_keyframes.append(frame_idx)
        
        if not valid_keyframes:
            print("[Warning] No valid keyframes to extract")
            return
        
        # Create temporary file for 720p video
        temp_dir = tempfile.mkdtemp()
        temp_video_path = os.path.join(temp_dir, "temp_720p.mp4")
        
        try:
            # Step 1: Pre-process video to 720p
            processed_video_path, target_width, target_height, was_processed = TransNetV2Torch.preprocess_video_to_720p(
                video_path, temp_video_path
            )
            
            # Step 2: Extract keyframes from processed video
            TransNetV2Torch.extract_keyframes_from_preprocessed_video(
                processed_video_path, valid_keyframes, output_dir, fps, target_width, target_height
            )
            
            print(f"[TransNetV2] Completed! Keyframes saved to {output_dir}")
            
        finally:
            # Clean up temporary files
            try:
                if os.path.exists(temp_video_path):
                    os.remove(temp_video_path)
                os.rmdir(temp_dir)
            except:
                pass  # Ignore cleanup errors

    @staticmethod
    def extract_keyframes_from_preprocessed_video(video_path: str, keyframe_indices, output_dir, fps, target_width, target_height):
        """
        Extract keyframes from pre-processed 720p video (much faster)
        
        Args:
            video_path: Path to pre-processed video
            keyframe_indices: List of frame indices to extract
            output_dir: Directory to save keyframes
            fps: Video FPS
            target_width: Video width
            target_height: Video height
        """
        import os
        from PIL import Image
        
        print(f"[TransNetV2] Extracting {len(keyframe_indices)} keyframes from pre-processed video")
        
        for i, frame_idx in enumerate(keyframe_indices):
            try:
                # Calculate timestamp
                timestamp = frame_idx / fps
                
                # Extract frame at original resolution (no additional scaling needed)
                video_stream, _ = ffmpeg.input(video_path, ss=timestamp).output(
                    "pipe:", 
                    vframes=1, 
                    format="rawvideo", 
                    pix_fmt="rgb24"
                ).run(capture_stdout=True, capture_stderr=True, quiet=True)
                
                # Check if we got data
                if len(video_stream) == 0:
                    print(f"\n[Warning] No data for frame {frame_idx} at timestamp {timestamp:.2f}s, skipping")
                    continue
                
                # Calculate expected size
                expected_size = target_width * target_height * 3
                if len(video_stream) != expected_size:
                    print(f"\n[Warning] Frame {frame_idx}: expected {expected_size} bytes, got {len(video_stream)}, skipping")
                    continue
                
                # Reshape frame
                frame_array = np.frombuffer(video_stream, np.uint8).reshape([target_height, target_width, 3])
                
                # Convert to PIL Image and save as JPG
                img = Image.fromarray(frame_array)
                
                # Generate filename: <frame_idx>.jpg
                filename = f"{frame_idx:06d}.jpg"
                filepath = os.path.join(output_dir, filename)
                img.save(filepath, 'JPEG', quality=90, optimize=True)
                
                print(f"\r[TransNetV2] Saved: {filename} ({i+1}/{len(keyframe_indices)})", end="")
                
            except Exception as e:
                print(f"\n[Error] extracting frame {frame_idx}: {e}")
                continue
        
        print(f"\n[TransNetV2] Keyframe extraction completed!")

    @staticmethod
    def calculate_720p_resolution(original_width, original_height):
        """
        Calculate target resolution to scale video to 720p while maintaining aspect ratio
        Only scales down if original resolution is higher than 720p
        
        Args:
            original_width: Original video width
            original_height: Original video height
            
        Returns:
            tuple: (target_width, target_height, needs_scaling) for 720p scaling
        """
        # If video is already 720p or smaller, don't scale
        if original_height <= 720:
            return original_width, original_height, False
            
        # Scale down to 720p
        target_height = 720
        aspect_ratio = original_width / original_height
        target_width = int(target_height * aspect_ratio)
        
        # Ensure width is even (required for some video encoders)
        if target_width % 2 != 0:
            target_width += 1
            
        return target_width, target_height, True

    @staticmethod
    def preprocess_video_to_720p(video_path: str, output_path: str):
        """
        Pre-process video to 720p resolution for faster keyframe extraction
        
        Args:
            video_path: Path to original video
            output_path: Path to save 720p video
            
        Returns:
            tuple: (processed_video_path, target_width, target_height, needs_processing)
        """
        import tempfile
        
        try:
            # Get video info
            probe = ffmpeg.probe(video_path)
            video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
            original_width = int(video_info['width'])
            original_height = int(video_info['height'])
            
            # Check if processing is needed
            target_width, target_height, needs_scaling = TransNetV2Torch.calculate_720p_resolution(
                original_width, original_height
            )
            
            if not needs_scaling:
                print(f"[TransNetV2] Video already 720p or smaller ({original_width}x{original_height}), using original")
                return video_path, original_width, original_height, False
            
            # Create temporary 720p video
            print(f"[TransNetV2] Pre-processing video: {original_width}x{original_height} → {target_width}x{target_height}")
            
            # Fast video processing with hardware acceleration if available
            stream = ffmpeg.input(video_path)
            stream = ffmpeg.output(
                stream, 
                output_path,
                vcodec='libx264',  # Fast encoder
                crf=23,  # Good quality/speed balance
                preset='fast',  # Fast encoding preset
                s=f"{target_width}x{target_height}",
                sws_flags="bilinear",  # Fast scaling
                y=None  # Overwrite if exists
            )
            
            ffmpeg.run(stream, capture_stdout=True, capture_stderr=True, quiet=True)
            
            print(f"[TransNetV2] Video pre-processing completed: {output_path}")
            return output_path, target_width, target_height, True
            
        except Exception as e:
            print(f"[Error] Video pre-processing failed: {e}")
            # Return original video if processing fails
            return video_path, original_width, original_height, False


# ------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path",
                        help="video file or directory containing videos")
    parser.add_argument("output_path", 
                        help="directory to save keyframes")
    parser.add_argument("--shot-threshold", type=float, default=0.35,
                        help="threshold for scene detection (default: 0.35)")
    parser.add_argument("--filter-range", type=str, default=None,
                        help="filter videos by number range (e.g., '08-10' for videos 08, 09, 10)")
    args = parser.parse_args()

    model = TransNetV2Torch()

    video_ext = (".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv")

    # Determine input files
    if os.path.isdir(args.input_path):
        files = [
            os.path.join(args.input_path, f)
            for f in os.listdir(args.input_path)
            if f.lower().endswith(video_ext)
        ]
    else:
        files = [args.input_path]

    if not files:
        print("No video files found!")
        return

    pbar = tqdm(files)
    for fp in pbar:
        video_name = os.path.splitext(os.path.basename(fp))[0]
        pbar.set_description(f"Processing {video_name}")

        # Create output directory for this video
        video_output_dir = os.path.join(args.output_path, video_name)
        
        # Skip if already processed (check if folder exists and has files)
        if os.path.exists(video_output_dir) and os.listdir(video_output_dir):
            print(f"\n[SKIP] {video_name} already processed. Folder exists with files.")
            continue

        try:
            # Process video
            frames, s_pred, a_pred = model.predict_video(fp)

            # Extract keyframes from scene middle points
            keyframe_indices = model.extract_keyframes_from_scenes(
                s_pred, threshold=args.shot_threshold
            )
            
            if keyframe_indices:
                # Save keyframes to folder
                model.save_keyframes_to_folder(
                    fp, keyframe_indices, video_output_dir
                )
                
                # Print summary
                print(f"[Summary] {video_name}: {len(keyframe_indices)} scenes = {len(keyframe_indices)} keyframes")
            else:
                print(f"\n[Warning] No scenes/keyframes found for {video_name}")
                
        except Exception as e:
            print(f"\n[Error] Failed to process {video_name}: {e}")
            continue


if __name__ == "__main__":
    main()
