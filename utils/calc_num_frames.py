import re
import sys

def parse_partition_string(s):
    # Example: 'c1b3t9' -> {'condition': 1, 'buffer': 3, 'target': 9}
    pattern = r"c(\d+)b(\d+)t(\d+)"
    m = re.match(pattern, s)
    if not m:
        raise ValueError(f"Invalid partition string: {s}")
    return {
        'condition': int(m.group(1)),
        'buffer': int(m.group(2)),
        'target': int(m.group(3)),
    }

def calc_frames(partition):
    # First latent (condition) is 1 frame, rest are 4 frames per latent
    cond_latents = partition['condition']
    buffer_latents = partition['buffer']
    target_latents = partition['target']
    # condition: first latent is 1 frame, rest are 4 frames each
    if cond_latents == 0:
        cond_frames = 0
    elif cond_latents == 1:
        cond_frames = 1
    else:
        cond_frames = 1 + (cond_latents - 1) * 4
    buffer_frames = buffer_latents * 4
    target_frames = target_latents * 4
    total_frames = cond_frames + buffer_frames + target_frames
    return cond_latents, cond_frames, buffer_latents, buffer_frames, target_latents, target_frames, total_frames

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python calc_num_frames.py <partition_string>")
        print("Example: python calc_num_frames.py c1b3t9")
        sys.exit(1)
    partition_string = sys.argv[1]
    partition = parse_partition_string(partition_string)
    cond_latents, cond_frames, buffer_latents, buffer_frames, target_latents, target_frames, total_frames = calc_frames(partition)
    print(f"condition: {cond_latents} (frames: {cond_frames})")
    print(f"buffer: {buffer_latents} (frames: {buffer_frames})")
    print(f"target: {target_latents} (frames: {target_frames})")
    print(f"total frames: {total_frames}")
