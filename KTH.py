import imageio
from tqdm import tqdm
import os
import pickle
import re
from torchvision import transforms
import torch

CATEGORIES = [
    "boxing",
    "handclapping",
    "handwaving",
    "jogging",
    "running",
    "walking"
]

base_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((80, 60)),
    transforms.Grayscale(num_output_channels=1)
])

def make_raw_dataset(directory="data", transform=None, f=9):
    """
    Make a raw dataset(format: {subject_id}.p) into 'data' directory from the raw KTH video dataset.
    Dataset are divided according to the instruction at:
    http://www.nada.kth.se/cvap/actions/00sequences.txt
    """
    frames_idx = parse_sequence_file()
    if not transform :
        transform = base_transform

    subjects = 25
    data = [[] for _ in range(subjects)] # list of data of each subject (total 25 subjects(people))
    raw_path = os.path.join(os.getcwd(), "kth")  # directory path that the KTH dataset videos are stored
    dir_path = os.path.join(os.getcwd(), directory) # directory path that the processed dataset will be stored
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    print("Processing ...")
    for category in CATEGORIES:
        # Get all files in current category's folder.
        folder_path = os.path.join(raw_path, category)
        filenames = sorted(os.listdir(folder_path))

        for filename in tqdm(filenames, desc=category):
            file_path = os.path.join(folder_path, filename)

            # Get id of person in this video.
            subject_id = int(filename.split("_")[0][6:])-1

            vid = imageio.get_reader(file_path, "ffmpeg")
            input = []
            seg_frames = []

            # Add each frame to correct list.
            seg_cnt = 0
            seg = frames_idx[filename][seg_cnt]
            
            for i, frame in enumerate(vid):
                if i < seg[0]:
                    continue
                elif i <= seg[1]:
                    frame = transform(frame).to(device)
                    if len(seg_frames) == 0:
                        seg_frames = frame
                        seg_frames.to(device)
                    else:
                        seg_frames = torch.cat([seg_frames, frame], dim=0) 
                    
                    if i == seg[1]: # this is last frame of this segment 
                        N, throw = seg_frames.shape[0] // f, seg_frames.shape[0] % f
                        if throw > 0:
                            seg_frames = seg_frames[:-throw]
                        seg_frames = seg_frames.reshape(N, f, seg_frames.shape[-2], seg_frames.shape[-1])
                        
                        if seg_cnt == 0:
                            input = seg_frames[:]
                            input.to(device)
                        else:
                            input = torch.cat([input, seg_frames], dim=0) 
                        
                        # segment update
                        seg_cnt += 1
                        if seg_cnt >= len(frames_idx[filename]): # no more segment
                            break
                        else:
                            seg_frames = []
                            seg = frames_idx[filename][seg_cnt]

            data[subject_id].append({
                "filename": filename,
                "category": category,
                "input": input, # Tensor shape : (N, f, c, h, w)
                "subject": subject_id
            })

    # save the data per each subject
    print("\nData Saving ...")
    for subject_id, d in enumerate(tqdm(data, desc="saving")): 
        person_path = os.path.join(dir_path, str(subject_id))
        pickle.dump(d, open("%s.p" % person_path, "wb"))
    

def parse_sequence_file():
    print("Parsing ./kth/00sequences.txt")

    # Read 00sequences.txt file.
    with open('./kth/00sequences.txt', 'r') as content_file:
        content = content_file.read()

    # Replace tab and newline character with space, then split file's content
    # into strings.
    content = re.sub("[\t\n]", " ", content).split()

    # Dictionary to keep ranges of frames with humans.
    # Example:
    # video "person01_boxing_d1": [(1, 95), (96, 185), (186, 245), (246, 360)].
    frames_idx = {}

    # Current video that we are parsing.
    current_filename = ""

    for s in content:
        if s == "frames":
            # Ignore this token.
            continue
        elif s.find("-") >= 0:
            # This is the token we are looking for. e.g. 1-95.
            if s[len(s) - 1] == ',':
                # Remove comma.
                s = s[:-1]

            # Split into 2 numbers => [1, 95]
            idx = s.split("-")

            # Add to dictionary.
            if not current_filename in frames_idx:
                frames_idx[current_filename] = []
            frames_idx[current_filename].append((int(idx[0]), int(idx[1])))
        else:
            # Parse next file.
            current_filename = s + "_uncomp.avi"

    return frames_idx

if __name__ == "__main__":
    print("Making dataset")
    make_raw_dataset()