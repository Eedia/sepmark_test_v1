
import ffmpeg_utils_e as f

video_path = f"./video/sample.mp4"
output_dir="./test_output_encoded"

print(video_path)
# f.extract_frames(video_path)
f.combine_frames(output_dir, video_path)
