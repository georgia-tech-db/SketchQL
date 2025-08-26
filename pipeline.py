import os
import torch
from data.video_data import read_primitives, read_primitives_old, read_primitives_topview_soccer
from data.visual_query import VisualQuery, generate_visual_query
from search.search import execute_qeury_multi_obj_sim_search_learned_model
from data.visualization import write_clips, visualize_query
from data.utils import stretch_objs, remove_overlap_clips_obj

dataset = "traffic"
query_name = "car_left_turn"#carstop_personwalk
model_cp_path = "data/model_checkpoint/model_cp.pt"


if dataset == "topview_soccer":
    video_file = "data/videos/topview_soccer.mp4"
    tracking_results_file = "data/primitives/topview_soccer.pkl"
    objs_in_video_dict, objs_at_frame, centroids_at_frame = read_primitives_topview_soccer(tracking_results_file)
elif dataset == "soccer":
    video_file = "data/videos/soccer-train-concat.mp4"
    tracking_results_file = "data/primitives/soccer-train-concat.mp4.pkl"
    objs_in_video_dict, objs_at_frame, centroids_at_frame = read_primitives(tracking_results_file)
elif dataset == "traffic":
    video_file = "data/videos/VIRAT_S_050300_01_000148_000396.mp4"
    tracking_results_file = "data/primitives/VIRAT_S_050300_01_000148_000396.mp4.pkl"
    objs_in_video_dict, objs_at_frame = read_primitives_old(tracking_results_file, width=1920, height=1080)
elif dataset == "bdd100k":
    video_file = "data/videos/bdd100k.mp4"
    tracking_results_file = "data/primitives/bdd100k.mp4.pkl"
    objs_in_video_dict, objs_at_frame = read_primitives_old(tracking_results_file,  width=1280, height=720)
elif dataset == "football_ytb":
    video_file = "data/videos/football_ytb.mp4"
    tracking_results_file = "data/primitives/football_ytb.mp4.pkl"
    objs_in_video_dict, objs_at_frame = read_primitives_old(tracking_results_file,  width=1280, height=720)

centroids_at_frame = None
query_objs_dict, relation = generate_visual_query(query_name)

query_objs_dict = stretch_objs(query_objs_dict, ratio=1.5)
visualize_query(query_objs_dict, query_name)
vquery = VisualQuery(query_objs_dict, relation)

if torch.cuda.is_available() and torch.cuda.device_count() > 0:
    device = torch.device(f'cuda:{0}')  # Use first GPU
else:
    device = torch.device('cpu')

exe_results, all_features, probs, all_cand_seq_centroids = execute_qeury_multi_obj_sim_search_learned_model(
    objs_in_video_dict, objs_at_frame, centroids_at_frame, vquery, model_cp_path, device)

print("Total matches after similarity search: %d"%len(exe_results))

exe_results, probs = remove_overlap_clips_obj(exe_results, probs)
print("Total matches after removing temporal overlaps: %d"%len(exe_results))

exe_results = [x for _, x in sorted(zip(probs.tolist(), exe_results), key=lambda pair: -pair[0])]

#visualize search results
write_clips(exe_results, video_file, n=60, folder=os.path.join(dataset, query_name, "learned_model"))
