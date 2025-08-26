# SketchQL: Zero-Shot Video Moment Retrieval with Sketch-Based Queries

## Overview
Sketch-QL is a video database management system for retrieving video moments with a sketch-based query interface.This interface allows users to specify object trajectory events with simple mouse drag-and-drop operations. Using a pre-trained model that encodes trajectory similarity, Sketch-QL achieves zero-shot video moments retrieval by performing similarity searches over the video to identify clips that are the most similar to the visual query.

## Installation

### Setup Environment
```
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```
### Install FFmpeg (for video processing)
```
# MacOS
brew install ffmpeg

# Ubuntu/Debian
apt install ffmpeg
```

### Data Preparation
1. Download Dataset: Download the traffic dataset from [Google Drive](https://drive.google.com/file/d/1DIy0NOBPTnRaDsnqSl-o1e3vAeFfM3Kz/view?usp=sharing) and place it in the `data/videos/` folder.


2. Download Model Checkpoint: Download the pre-trained model from [Google Drive](https://drive.google.com/file/d/1DIy0NOBPTnRaDsnqSl-o1e3vAeFfM3Kz/view?usp=sharing) and place it in the `data/model_checkpoint` folder.

## Usage

### Running the pipepline
```
# Run the main pipeline
python3 pipeline.py
```
- Sample queries: Located at `output/queries/`
- Results: Retrieved video clips are saved under `output/query_results/`


## Citation
If you use SketchQL in your research, please cite our work as follows:
```
@article{sketchql,
  author = {Wu, Renzhi and Chunduri, Pramod and Payani, Ali and Chu, Xu and Arulraj, Joy and Rong, Kexin},
  title = {SketchQL: Video Moment Querying with a Visual Query Interface},
  year = {2024},
  volume = {2},
  number = {4},
  doi = {10.1145/3677140},
  journal = {Proc. ACM Manag. Data},
  month = sep,
  articleno = {204},
  numpages = {27}
}

@article{sketchql-demo,
  author = {Wu, Renzhi and Chunduri, Pramod and Shah, Dristi J and Aravind, Ashmitha Julius and Payani, Ali and Chu, Xu and Arulraj, Joy and Rong, Kexin},
  title = {SketchQL Demonstration: Zero-Shot Video Moment Querying with Sketches},
  year = {2024},
  volume = {17},
  number = {12},
  issn = {2150-8097},
  doi = {10.14778/3685800.3685892},
  journal = {Proc. VLDB Endow.},
  month = aug,
  pages = {4429–4432},
  numpages = {4}
}
```
