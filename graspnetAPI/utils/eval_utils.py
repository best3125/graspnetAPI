__author__ = 'cxwang, mhgou'
__version__ = '1.0'

from collections import defaultdict
from copy import deepcopy
import os
import time
from typing import List
import numpy as np
import open3d as o3d
from transforms3d.euler import euler2mat, quat2mat

from graspnetAPI.grasp import GraspGroup

from .rotation import batch_viewpoint_params_to_matrix, matrix_to_dexnet_params

from .dexnet.grasping.quality import PointGraspMetrics3D
from .dexnet.grasping.grasp import ParallelJawPtGrasp3D
from .dexnet.grasping.graspable_object import GraspableObject3D
from .dexnet.grasping.grasp_quality_config import GraspQualityConfigFactory
from .dexnet.grasping.contacts import Contact3D
from .dexnet.grasping.meshpy.obj_file import ObjFile
from .dexnet.grasping.meshpy.sdf_file import SdfFile

def get_scene_name(num):
    '''
    **Input:**
    - num: int of the scene number.

    **Output:**
    - string of the scene name.
    '''
    return ('scene_%04d' % (num,))

def create_table_points(lx, ly, lz, dx=0, dy=0, dz=0, grid_size=0.01):
    '''
    **Input:**
    - lx:
    - ly:
    - lz:
    **Output:**
    - numpy array of the points with shape (-1, 3).
    '''
    xmap = np.linspace(0, lx, int(lx/grid_size))
    ymap = np.linspace(0, ly, int(ly/grid_size))
    zmap = np.linspace(0, lz, int(lz/grid_size))
    xmap, ymap, zmap = np.meshgrid(xmap, ymap, zmap, indexing='xy')
    xmap += dx
    ymap += dy
    zmap += dz
    points = np.stack([xmap, ymap, zmap], axis=-1)
    points = points.reshape([-1, 3])
    return points

def parse_posevector(posevector):
    '''
    **Input:**
    - posevector: list of pose
    **Output:**
    - obj_idx: int of the index of object.
    - mat: numpy array of shape (4, 4) of the 6D pose of object.
    '''
    mat = np.zeros([4,4],dtype=np.float32)
    alpha, beta, gamma = posevector[4:7]
    alpha = alpha / 180.0 * np.pi
    beta = beta / 180.0 * np.pi
    gamma = gamma / 180.0 * np.pi
    mat[:3,:3] = euler2mat(alpha, beta, gamma)
    mat[:3,3] = posevector[1:4]
    mat[3,3] = 1
    obj_idx = int(posevector[0])
    return obj_idx, mat

def load_dexnet_model(data_path):
    '''
    **Input:**

    - data_path: path to load .obj & .sdf files

    **Output:**
    - obj: dexnet model
    '''
    of = ObjFile('{}.obj'.format(data_path))
    sf = SdfFile('{}.sdf'.format(data_path))
    mesh = of.read()
    sdf = sf.read()
    obj = GraspableObject3D(sdf, mesh)
    return obj

def transform_points(points, trans):
    '''
    **Input:**

    - points: (N, 3)

    - trans: (4, 4)

    **Output:**
    - points_trans: (N, 3)
    '''
    ones = np.ones([points.shape[0],1], dtype=points.dtype)
    points_ = np.concatenate([points, ones], axis=-1)
    points_ = np.matmul(trans, points_.T).T
    points_trans = points_[:,:3]
    return points_trans

def compute_point_distance(A, B):
    '''
    **Input:**
    - A: (N, 3)

    - B: (M, 3)

    **Output:**
    - dists: (N, M)
    '''
    A = A[:, np.newaxis, :]
    B = B[np.newaxis, :, :]
    dists = np.linalg.norm(A-B, axis=-1)
    return dists

def compute_closest_points(A, B):
    '''
    **Input:**

    - A: (N, 3)

    - B: (M, 3)

    **Output:**

    - indices: (N,) closest point index in B for each point in A
    '''
    dists = compute_point_distance(A, B)
    indices = np.argmin(dists, axis=-1)
    return indices

def voxel_sample_points(points, voxel_size=0.008):
    '''
    **Input:**

    - points: (N, 3)

    **Output:**

    - points: (n, 3)
    '''
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud = cloud.voxel_down_sample(voxel_size)
    points = np.array(cloud.points)
    return points

def topk_grasps(grasps, k=10):
    '''
    **Input:**

    - grasps: (N, 17)

    - k: int

    **Output:**

    - topk_grasps: (k, 17)
    '''
    assert(k > 0)
    grasp_confidence = grasps[:, 0]
    indices = np.argsort(-grasp_confidence)
    topk_indices = indices[:min(k, len(grasps))]
    topk_grasps = grasps[topk_indices]
    return topk_grasps

def get_grasp_score(grasp, obj, fc_list, force_closure_quality_config):
    # fc_list contains the friction coefficients
    # in descending order
    tmp, is_force_closure = False, False
    # in the end quality equals the lowest possible friction value for a force closure
    # if no force closure is possible -1 is returned
    quality = -1
    # for each friction coefficient
    for ind_, value_fc in enumerate(fc_list):
        value_fc = round(value_fc, 2)
        tmp = is_force_closure
        # is it a force closure for this friction coeff?
        is_force_closure = PointGraspMetrics3D.grasp_quality(grasp, obj, force_closure_quality_config[value_fc])
        # break if it was a force closure for last friction coeff and now it isnt anymore
        if tmp and not is_force_closure:
            quality = round(fc_list[ind_ - 1], 2)
            break
        # last friction coefficient did work
        elif is_force_closure and value_fc == fc_list[-1]:
            quality = value_fc
            break
        # break if it doesnt work with highest friction coefficient
        elif value_fc == fc_list[0] and not is_force_closure:
            break

    return quality

def collision_detection(grasp_list, model_list, dexnet_models, poses, scene_points,
                        outlier=0.05, empty_thresh=10, return_dexgrasps=False, fill_seperated_masks=False):
    '''
    **Input:**

    - grasp_list: [(k1, 17), (k2, 17), ..., (kn, 17)] in camera coordinate

    - model_list: [(N1, 3), (N2, 3), ..., (Nn, 3)] in camera coordinate

    - dexnet_models: [GraspableObject3D,] in object coordinate

    - poses: [(4, 4),] from model coordinate to camera coordinate

    - scene_points: (Ns, 3) in camera coordinate

    - outlier: float, used to compute workspace mask

    - empty_thresh: int, 'num_inner_points < empty_thresh' means empty grasp

    - return_dexgrasps: bool, return grasps in dex-net format while True

    **Output:**

    - collsion_mask_list: [(k1,), (k2,), ..., (kn,)]

    - empty_mask_list: [(k1,), (k2,), ..., (kn,)]

    - dexgrasp_list: [[ParallelJawPtGrasp3D,],] in object coordinate
    '''
    height = 0.02
    depth_base = 0.02
    finger_width = 0.01
    collision_mask_list = list()
    seperated_collision_mask_list = list()
    num_models = len(model_list)
    empty_mask_list = list()
    dexgrasp_list = list()

    for i in range(num_models):
        if len(grasp_list[i]) == 0:
            collision_mask_list.append(list())
            seperated_collision_mask_list.append(list())
            empty_mask_list.append(list())
            if return_dexgrasps:
                dexgrasp_list.append(list())
            continue

        ## parse grasp parameters
        model = model_list[i]
        obj_pose = poses[i]
        dexnet_model = dexnet_models[i]
        grasps = grasp_list[i]
        grasp_points = grasps[:, 13:16]
        grasp_poses = grasps[:, 4:13].reshape([-1,3,3])
        grasp_depths = grasps[:, 3]
        grasp_widths = grasps[:, 1]

        ## crop scene, remove outlier
        xmin, xmax = model[:,0].min(), model[:,0].max()
        ymin, ymax = model[:,1].min(), model[:,1].max()
        zmin, zmax = model[:,2].min(), model[:,2].max()
        xlim = ((scene_points[:,0] > xmin-outlier) & (scene_points[:,0] < xmax+outlier))
        ylim = ((scene_points[:,1] > ymin-outlier) & (scene_points[:,1] < ymax+outlier))
        zlim = ((scene_points[:,2] > zmin-outlier) & (scene_points[:,2] < zmax+outlier))
        workspace = scene_points[xlim & ylim & zlim]

        # transform scene to gripper frame
        target = (workspace[np.newaxis,:,:] - grasp_points[:,np.newaxis,:])
        target = np.matmul(target, grasp_poses)

        # compute collision mask
        mask1 = ((target[:,:,2]>-height/2) & (target[:,:,2]<height/2))
        mask2 = ((target[:,:,0]>-depth_base) & (target[:,:,0]<grasp_depths[:,np.newaxis]))
        mask3 = (target[:,:,1]>-(grasp_widths[:,np.newaxis]/2+finger_width))
        mask4 = (target[:,:,1]<-grasp_widths[:,np.newaxis]/2)
        mask5 = (target[:,:,1]<(grasp_widths[:,np.newaxis]/2+finger_width))
        mask6 = (target[:,:,1]>grasp_widths[:,np.newaxis]/2)
        mask7 = ((target[:,:,0]>-(depth_base+finger_width)) & (target[:,:,0]<-depth_base))

        left_mask = (mask1 & mask2 & mask3 & mask4)
        right_mask = (mask1 & mask2 & mask5 & mask6)
        bottom_mask = (mask1 & mask3 & mask5 & mask7)
        inner_mask = (mask1 & mask2 &(~mask4) & (~mask6))
        collision_mask = np.any((left_mask | right_mask | bottom_mask), axis=-1)
        empty_mask = (np.sum(inner_mask, axis=-1) < empty_thresh)
        collision_mask = (collision_mask | empty_mask)
        collision_mask_list.append(collision_mask)
        if fill_seperated_masks:
            seperated_collision_mask_list.append([
                    np.any(left_mask, axis=-1),
                    np.any(right_mask, axis=-1),
                    np.any(bottom_mask, axis=-1),
                    np.any(inner_mask, axis=-1)
                ])
        empty_mask_list.append(empty_mask)

        ## generate grasps in dex-net format
        if return_dexgrasps:
            dexgrasps = list()
            for grasp_id,_ in enumerate(grasps):
                grasp_point = grasp_points[grasp_id]
                R = grasp_poses[grasp_id]
                width = grasp_widths[grasp_id]
                depth = grasp_depths[grasp_id]
                points_in_gripper = target[grasp_id][inner_mask[grasp_id]]
                if empty_mask[grasp_id]:
                    dexgrasps.append(None)
                    continue
                center = np.array([depth, 0, 0]).reshape([3, 1]) # gripper coordinate
                center = np.dot(grasp_poses[grasp_id], center).reshape([3])
                center = (center + grasp_point).reshape([1,3]) # camera coordinate
                center = transform_points(center, np.linalg.inv(obj_pose)).reshape([3]) # object coordinate
                R = np.dot(obj_pose[:3,:3].T, R)
                binormal, approach_angle = matrix_to_dexnet_params(R)
                grasp = ParallelJawPtGrasp3D(ParallelJawPtGrasp3D.configuration_from_params(
                                            center, binormal, width, approach_angle), depth)
                dexgrasps.append(grasp)
            dexgrasp_list.append(dexgrasps)

    if return_dexgrasps:
        return collision_mask_list, empty_mask_list, dexgrasp_list, seperated_collision_mask_list
    else:
        return collision_mask_list, empty_mask_list, seperated_collision_mask_list

def eval_grasp(grasp_group, models, dexnet_models, poses, config, table=None, voxel_size=0.008, TOP_K = 50, fill_seperated_masks=True):
    '''
    **Input:**

    - grasp_group: GraspGroup instance for evaluation.

    - models: in model coordinate

    - dexnet_models: models in dexnet format

    - poses: from model to camera coordinate

    - config: dexnet config.

    - table: in camera coordinate

    - voxel_size: float of the voxel size.

    - TOP_K: int of the number of top grasps to evaluate.
    '''
    num_models = len(models)

    # construct scene
    model_trans_list = list()
    seg_mask = list()
    for i,model in enumerate(models):
        model_trans = transform_points(model, poses[i])
        seg = i * np.ones(model_trans.shape[0], dtype=np.int32)
        model_trans_list.append(model_trans)
        seg_mask.append(seg)
    seg_mask = np.concatenate(seg_mask, axis=0)
    scene = np.concatenate(model_trans_list, axis=0)

    # if remove_colliding_grasps_before_nms:
    # get indices of models that are nearest to respective grasp
    indices = compute_closest_points(grasp_group.translations, scene)
    model_to_grasp = seg_mask[indices]

    sort_idx_list = []

    grasp_list = sort_grasps_by_model(
        grasp_group=grasp_group,
        TOP_K=len(grasp_group), num_models=num_models, eval_all=True, model_to_grasp=model_to_grasp, sort_idx_list=sort_idx_list)

    ## collision detection
    if table is not None:
        scene_w_table = np.concatenate([scene, table])
    else:
        scene_w_table = scene

    collision_mask_list, empty_list, dexgrasp_list, seperated_collision_mask_list = collision_detection(
        grasp_list, model_trans_list, dexnet_models, poses, scene_w_table, outlier=0.05, return_dexgrasps=True, fill_seperated_masks=fill_seperated_masks)

    collision_mask_list = [l for l in collision_mask_list if len(l) != 0]
    collision_mask = np.concatenate(collision_mask_list)

    # create new grasp group where colliding grasps are filtered
    grasp_group_collision_filtered = deepcopy(grasp_group)
    grasp_group_collision_filtered = grasp_group_collision_filtered[sort_idx_list]
    grasp_group_collision_filtered.remove(collision_mask)

    res = {}

    ## evaluate grasps
    # score configurations
    force_closure_quality_config = dict()
    fc_list = np.array([1.2, 1.0, 0.8, 0.6, 0.4, 0.2])
    for value_fc in fc_list:
        value_fc = round(value_fc, 2)
        config['metrics']['force_closure']['friction_coef'] = value_fc
        force_closure_quality_config[value_fc] = GraspQualityConfigFactory.create_config(config['metrics']['force_closure'])

    for key, gg in zip(['unfiltered', 'collision_filtered'], [grasp_group, grasp_group_collision_filtered]):
        if TOP_K is None:
            TOP_K = len(grasp_group)
            eval_all = True
        else:
            eval_all = False
            ## grasp nms
            gg = gg.nms(0.03, 30.0/180*np.pi)
            # grasp_group_collision_filtered = grasp_group_collision_filtered.nms(0.03, 30.0/180*np.pi)

        # assign grasps
        # get indices of models that are nearest to respective grasp
        start = time.time()
        indices = compute_closest_points(gg.translations, scene)
        tmp = time.time()
        print('CP calc', tmp - start)
        start = tmp
        model_to_grasp = seg_mask[indices]

        sort_idx_list = list()

        grasp_list = sort_grasps_by_model(gg, TOP_K, num_models, eval_all, model_to_grasp, sort_idx_list)

        collision_mask_list, empty_list, dexgrasp_list, seperated_collision_mask_list = collision_detection(
                grasp_list, model_trans_list, dexnet_models, poses, scene_w_table, outlier=0.05, return_dexgrasps=True, fill_seperated_masks=fill_seperated_masks)
        tmp = time.time()
        print('coll detection', tmp - start)
        start = tmp

        # get grasp scores
        score_list = list()

        for i in range(num_models):
            dexnet_model = dexnet_models[i]
            collision_mask = collision_mask_list[i]
            dexgrasps = dexgrasp_list[i]
            scores = list()
            num_grasps = len(dexgrasps)
            for grasp_id in range(num_grasps):
                if collision_mask[grasp_id]:
                    scores.append(-1.)
                    continue
                if dexgrasps[grasp_id] is None:
                    scores.append(-1.)
                    continue
                grasp = dexgrasps[grasp_id]
                if grasp is None:
                    scores.append(0)
                else:
                    score = get_grasp_score(grasp, dexnet_model, fc_list, force_closure_quality_config)
                    scores.append(score)
            score_list.append(scores)
        tmp = time.time()
        print('grasp score calc', tmp - start)
        start = tmp
        res[key] = {
            'grasp_list': grasp_list,
            'score_list': score_list,
            'collision_mask_list': collision_mask_list,
            'empty_list': empty_list,
            'sort_idx_list': sort_idx_list,
            'seperated_collision_mask_list': seperated_collision_mask_list
        }

    return res
    return grasp_list, score_list, collision_mask_list, empty_list, sort_idx_list, seperated_collision_mask_list

def sort_grasps_by_model(grasp_group, TOP_K, num_models, eval_all, model_to_grasp, sort_idx_list):
    pre_grasp_list = list()
    grasp_indexes = np.arange(len(grasp_group))

    for i in range(num_models):
        grasp_i = grasp_group[model_to_grasp==i]
        grasp_indexes_m = grasp_indexes[model_to_grasp==i]

        sort_idx = np.argsort(grasp_i.scores)[::-1]
        sort_idx_list.extend(grasp_indexes_m[sort_idx].tolist())

        grasp_i.sort_by_score()
        if eval_all:
            pre_grasp_list.append(grasp_i.grasp_group_array)
        else:
            pre_grasp_list.append(grasp_i[:10].grasp_group_array)
    if eval_all:
        grasp_list = pre_grasp_list
    else:
        all_grasp_list = np.vstack(pre_grasp_list)
        remain_mask = np.argsort(all_grasp_list[:,0])[::-1]
        grasp_list = []

        if len(remain_mask) == 0:
            for i in range(num_models):
                grasp_list.append([])
        else:
            min_score = all_grasp_list[remain_mask[min(TOP_K, len(remain_mask)) - 1],0]

            # only keep grasps that have a higher quality than the topkth best grasp
            # effetivly only keeping the top k grasps
            for i in range(num_models):
                remain_mask_i = pre_grasp_list[i][:,0] >= min_score
                grasp_list.append(pre_grasp_list[i][remain_mask_i])
    return grasp_list
