import numpy as np
from sklearn.cluster import DBSCAN
from collections import defaultdict

import random
import cv2

def get_colors(n):
    """
    Generate n visually distinct colors using HSV color space
    
    Args:
        n: Number of distinct colors needed
    Returns:
        List of RGB tuples with n distinct colors. Returns empty list if n <= 0.
    """
    if n <= 0:
        return []
        
    colors = []
    hue_partition = 179 // n  
    
    for i in range(n):
        # Create color with:
        # - Evenly spaced hue for maximum distinction
        # - High saturation (240) for vivid colors
        # - High value (220) for better visibility
        hue = int(i * hue_partition)
        saturation = 240
        value = 220
        
        hsv_color = np.uint8([[[hue, saturation, value]]])
        rgb_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
        colors.append(tuple(map(int, rgb_color[::-1])))
    
    return colors

def cluster_bboxes_with_ids(groups_status, bboxes, track_ids, max_group_id, eps=50, min_samples=2, threshold_overlap=0.5):
    """
    Cluster tracked objects based on their center points using DBSCAN
    
    Args:
        centers: List of center points [[x,y], ...]
        track_ids: List of tracking IDs corresponding to centers
        eps: DBSCAN epsilon parameter
        min_samples: DBSCAN min_samples parameter
        
    Returns:
        List of dicts with id, center, cluster label and color for each tracked object
    """
    if not bboxes or not track_ids or len(bboxes) != len(track_ids):
        raise ValueError("bboxes and track_ids must have same length and not be empty")
    
    centers = [(x + w / 2, y + h / 2, (2 * 3.14 * 180) / (w + h * 360) * 1000 + 3) for (x, y, w, h) in bboxes]
    # Convert to numpy array for DBSCAN
    X = np.array(centers)

    # Run DBSCAN clustering
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X)

    cluster_members = defaultdict(list)
    for label, tid in zip(labels, track_ids):
        if label != -1:
            cluster_members[label].append(tid)

    new_group_status = defaultdict(list)
    if len(groups_status) > 0:       
        for id_cluster, members in cluster_members.items():
            # Check if this group is inherited from an old group
            best_match = None
            best_overlap = 0
            overlap_ratio = 0
            # Find the old group with the largest overlap with new group
            for group_id, old_group_members in groups_status.items():
                overlap = len(set(members) & set(old_group_members))
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_match = group_id
                    overlap_ratio = best_overlap / len(old_group_members)

            if best_match is not None and overlap_ratio >= threshold_overlap:
                new_group_status[best_match] = cluster_members[id_cluster]
            else:
                # Create new group if no inheritance found
                new_group_status[max_group_id + 1] = cluster_members[id_cluster]
                max_group_id += 1
                
    else:
        new_group_status = cluster_members
    # Update max_group_id based on new groups
        max_group_id = max(new_group_status.keys(), default=max_group_id)
    # Generate colors for groups
    n_clusters = len(new_group_status)
    colors = get_colors(n_clusters)
    color_map = dict(zip(new_group_status.keys(), colors))
    #Map ids to cluster labels
    idp_to_idg_map = {id_p: id_g for id_g, id_ps in new_group_status.items() for id_p in id_ps}
    # Create results
    results = []
    for bbox, id_p, id_g in zip(bboxes, track_ids, labels):
        id_g = idp_to_idg_map.get(id_p, -1)
        color = (128, 128, 128) if id_g == -1 else color_map[id_g]

        results.append({
            'id_p': id_p,
            'bbox': bbox,
            'id_g': id_g,
            'color': color
        })
    return results, new_group_status, max_group_id
