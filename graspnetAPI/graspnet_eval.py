__author__ = 'mhgou, cxwang and hsfang'

from collections import defaultdict
from typing import Dict, Optional
import numpy as np
import os
import logging
import pickle
import open3d as o3d
import json
import time
import datetime

from .graspnet import GraspNet
from .grasp import GraspGroup
from .utils.config import get_config
from .utils.eval_utils import (
    get_scene_name,
    create_table_points,
    parse_posevector,
    load_dexnet_model,
    transform_points,
    compute_point_distance,
    compute_closest_points,
    voxel_sample_points,
    topk_grasps,
    get_grasp_score,
    collision_detection,
    eval_grasp
)
from .utils.xmlhandler import xmlReader
from .utils.utils import generate_scene_model

class GraspNetEval(GraspNet):
    '''
    Class for evaluation on GraspNet dataset.

    **Input:**

    - root: string of root path for the dataset.

    - camera: string of type of the camera.

    - split: string of the date split.
    '''
    def __init__(self, root, camera, split = 'test'):
        super(GraspNetEval, self).__init__(root, camera, split)

    def get_scene_models(self, scene_id, ann_id):
        '''
            return models in model coordinate
        '''
        model_dir = os.path.join(self.root, 'models')
        # print('Scene {}, {}'.format(scene_id, camera))
        scene_reader = xmlReader(os.path.join(self.root, 'scenes', get_scene_name(scene_id), self.camera, 'annotations', '%04d.xml' % (ann_id,)))
        posevectors = scene_reader.getposevectorlist()
        obj_list = []
        model_list = []
        dexmodel_list = []
        for posevector in posevectors:
            obj_idx, _ = parse_posevector(posevector)
            obj_list.append(obj_idx)
        for obj_idx in obj_list:
            model = o3d.io.read_point_cloud(os.path.join(model_dir, '%03d' % obj_idx, 'nontextured.ply'))
            dex_cache_path = os.path.join(self.root, 'dex_models', '%03d.pkl' % obj_idx)
            if os.path.exists(dex_cache_path):
                with open(dex_cache_path, 'rb') as f:
                    dexmodel = pickle.load(f)
            else:
                dexmodel = load_dexnet_model(os.path.join(model_dir, '%03d' % obj_idx, 'textured'))
            points = np.array(model.points)
            model_list.append(points)
            dexmodel_list.append(dexmodel)
        return model_list, dexmodel_list, obj_list


    def get_model_poses(self, scene_id, ann_id):
        '''
        **Input:**

        - scene_id: int of the scen index.

        - ann_id: int of the annotation index.

        **Output:**

        - obj_list: list of int of object index.

        - pose_list: list of 4x4 matrices of object poses.

        - camera_pose: 4x4 matrix of the camera pose relative to the first frame.

        - align mat: 4x4 matrix of camera relative to the table.
        '''
        scene_dir = os.path.join(self.root, 'scenes')
        camera_poses_path = os.path.join(self.root, 'scenes', get_scene_name(scene_id), self.camera, 'camera_poses.npy')
        camera_poses = np.load(camera_poses_path)
        camera_pose = camera_poses[ann_id]
        align_mat_path = os.path.join(self.root, 'scenes', get_scene_name(scene_id), self.camera, 'cam0_wrt_table.npy')
        align_mat = np.load(align_mat_path)
        # print('Scene {}, {}'.format(scene_id, camera))
        scene_reader = xmlReader(os.path.join(scene_dir, get_scene_name(scene_id), self.camera, 'annotations', '%04d.xml'% (ann_id,)))
        posevectors = scene_reader.getposevectorlist()
        obj_list = []
        pose_list = []
        for posevector in posevectors:
            obj_idx, mat = parse_posevector(posevector)
            obj_list.append(obj_idx)
            pose_list.append(mat)
        return obj_list, pose_list, camera_pose, align_mat


    def eval_scene(self, scene_id: int, dump_folder: str,
                   TOP_K: int = 50,
                   vis: bool = False, max_width:float = 0.1,
                   log_dict: Optional[Dict] = None):
        """Evaluated grasps loaded from dump_folder for specified scene.
        Additional information can be return through log_dict parameter.

        Parameters
        ----------
        scene_id : int
            Id of the scene to evaluate.
        dump_folder : str
            Folder root for loading grasps. Structure has to conform to <dump_folder>/scene_<scene_id>/<camera>/<annotation_id>.npy.
        TOP_K : int, optional
            Number of best grasps used for evaluation. Rest will be ignored, by default 50
        vis : bool, optional
            Wether to visualize single evaluations through open3d, by default False
        max_width : float, optional
            Maximum width of gripper. Grasps which are wider will be clipped, by default 0.1
        log_dict : dict, optional
            If a dict is supplied additional information will be recorded in this dictionary, by default None

        Returns
        -------
        numpy.ndarray
            Accuracies as numpy array with shape (256, TOP_K, 6). Fist dimension is number of annotations per scene and last ist number
            of tested friction coefficients.
        """
        #TODO: Make this code a bit more readable by refactoring
        config = get_config()
        table = create_table_points(1.0, 1.0, 0.05, dx=-0.5, dy=-0.5, dz=-0.05, grid_size=0.008)

        do_extensive_eval = TOP_K is None

        list_coe_of_friction = [0.2,0.4,0.6,0.8,1.0,1.2]

        model_list, dexmodel_list, _ = self.get_scene_models(scene_id, ann_id=0)

        model_sampled_list = list()
        for model in model_list:
            model_sampled = voxel_sample_points(model, 0.008)
            model_sampled_list.append(model_sampled)

        scene_accuracy = defaultdict(list)
        grasp_list_list = []
        score_list_list = []
        collision_list_list = []

        if log_dict is not None:
            # need to do update here otherwise a new dict is created and the reference
            # outside this function points to the wrong dict
            log_dict.update({"unfiltered": {
                "scene_id": scene_id,
                "TOP_K": TOP_K,
                "friction_coeffs": list_coe_of_friction,
                "max_width": max_width,
                "scores": {
                    "unfiltered": [],
                    "after_collision": [],
                    "force_closure_only": [],
                    "rejected": []
                },
                "confidences": {
                    "unfiltered": [],
                    "after_collision": [],
                    "force_closure_only": [],
                    "rejected": []
                },
                "empty_grasp_list_count": 0,
                "unfiltered_grasp_counts": [],
                "collision_counts": {
                    "combined": [],
                    "left": [],
                    "right": [],
                    "bottom": [],
                    "inner": []
                },
                "width_clip_counts": {
                    "min": [],
                    "max": []
                },
                "grasp_accuracies": [],
                "collision_lists": [],
                "score_lists": [],
                "sortidx_lists": []
            }, 'collision_filtered': {
                "scene_id": scene_id,
                "TOP_K": TOP_K,
                "friction_coeffs": list_coe_of_friction,
                "max_width": max_width,
                "scores": {
                    "unfiltered": [],
                    "after_collision": [],
                    "force_closure_only": [],
                    "rejected": []
                },
                "confidences": {
                    "unfiltered": [],
                    "after_collision": [],
                    "force_closure_only": [],
                    "rejected": []
                },
                "empty_grasp_list_count": 0,
                "unfiltered_grasp_counts": [],
                "collision_counts": {
                    "combined": [],
                    "left": [],
                    "right": [],
                    "bottom": [],
                    "inner": []
                },
                "width_clip_counts": {
                    "min": [],
                    "max": []
                },
                "grasp_accuracies": [],
                "collision_lists": [],
                "score_lists": [],
                "sortidx_lists": []
            }})

        for ann_id in range(256):
            start_timestamp = time.time()
            grasp_group = GraspGroup().from_npy(os.path.join(dump_folder,get_scene_name(scene_id), self.camera, '%04d.npy' % (ann_id,)))

            # TODO: Add warnings when grasps are obiously wrong (z-translation negative) as these should be filtered beforehand
            # grasp_group.remove(np.argwhere(grasp_group.translations[:, 2] <= 0))

            _, pose_list, camera_pose, align_mat = self.get_model_poses(scene_id, ann_id)
            table_trans = transform_points(table, np.linalg.inv(np.matmul(align_mat, camera_pose)))

            # clip width to [0,max_width]
            gg_array = grasp_group.grasp_group_array
            min_width_mask = (gg_array[:,1] < 0)
            max_width_mask = (gg_array[:,1] > max_width)
            gg_array[min_width_mask,1] = 0
            gg_array[max_width_mask,1] = max_width
            grasp_group.grasp_group_array = gg_array

            # if log_dict is not None:
            #     log_dict["width_clip_counts"]["min"].append(int(min_width_mask.sum()))
            #     log_dict["width_clip_counts"]["max"].append(int(max_width_mask.sum()))

            # grasp_list, score_list, collision_mask_list, empty_list, sort_idx_list, seperated_collision_mask_list = eval_grasp(
            eval_res = eval_grasp(
                grasp_group, model_sampled_list, dexmodel_list, pose_list, config,
                table=table_trans, voxel_size=0.008, TOP_K = None if do_extensive_eval else TOP_K, fill_seperated_masks=log_dict is not None)

            for key, e_res in eval_res.items():
                grasp_list = e_res['grasp_list']
                score_list = e_res['score_list']
                collision_mask_list = e_res['collision_mask_list']
                empty_list = e_res['empty_list']
                sort_idx_list = e_res['sort_idx_list']
                seperated_collision_mask_list = e_res['seperated_collision_mask_list']

                # remove empty
                grasp_list = [x for x in grasp_list if len(x) != 0]
                score_list = [x for x in score_list if len(x) != 0]
                if log_dict is not None:
                    seperated_collision_mask_list = [
                            seperated_collision_mask_list[idx] for idx, x in enumerate(collision_mask_list) if len(x)!=0
                        ]
                collision_mask_list = [x for x in collision_mask_list if len(x)!=0]

                if len(grasp_list) == 0:
                    # TODO: fix logging
                    # if log_dict: log_dict["empty_grasp_list_count"] += 1
                    grasp_accuracy = np.zeros((TOP_K,len(list_coe_of_friction)))
                    scene_accuracy[key].append(grasp_accuracy)
                    scene_accuracy[key + '_cf'].append(grasp_accuracy)
                    grasp_list_list.append([])
                    score_list_list.append([])
                    collision_list_list.append([])

                    print('\rMean Accuracy for scene:{} ann:{}='.format(scene_id, ann_id),np.mean(grasp_accuracy[:,:]))
                    continue

                # concat into scene level
                grasp_list, score_list, collision_mask_list = np.concatenate(grasp_list), np.concatenate(score_list), np.concatenate(collision_mask_list)

                if log_dict is not None:
                    left_list = []
                    right_list = []
                    bottom_list= []
                    inner_list = []
                    for x in seperated_collision_mask_list:
                        left, right, bottom, inner = x
                        left_list.append(left)
                        right_list.append(right)
                        bottom_list.append(bottom)
                        inner_list.append(inner)
                    seperated_collision_mask_list = [
                        np.concatenate(left_list) & collision_mask_list,
                        np.concatenate(right_list) & collision_mask_list,
                        np.concatenate(bottom_list) & collision_mask_list,
                        np.concatenate(inner_list) & collision_mask_list,
                    ]

                    log_dict[key]["unfiltered_grasp_counts"].append(len(grasp_list))
                    log_dict[key]["collision_counts"]["left"].append(int(np.sum(~seperated_collision_mask_list[0])))
                    log_dict[key]["collision_counts"]["right"].append(int(np.sum(~seperated_collision_mask_list[1])))
                    log_dict[key]["collision_counts"]["bottom"].append(int(np.sum(~seperated_collision_mask_list[2])))
                    log_dict[key]["collision_counts"]["inner"].append(int(np.sum(seperated_collision_mask_list[3])))

                    log_dict[key]["collision_counts"]["combined"].append(int(np.sum(~collision_mask_list)))

                    log_dict[key]["scores"]["unfiltered"].append(score_list.tolist())
                    log_dict[key]["confidences"]["unfiltered"].append(grasp_list[:,0].tolist())
                    log_dict[key]["scores"]["rejected"].append(score_list[collision_mask_list].tolist())
                    log_dict[key]["confidences"]["rejected"].append(grasp_list[collision_mask_list][:,0].tolist())

                if vis:
                    t = o3d.geometry.PointCloud()
                    t.points = o3d.utility.Vector3dVector(table_trans)
                    model_list = generate_scene_model(self.root, 'scene_%04d' % scene_id , ann_id, return_poses=False, align=False, camera=self.camera)
                    import copy
                    gg = GraspGroup(copy.deepcopy(grasp_list))
                    scores = np.array(score_list)
                    scores = scores / 4 + 0.75 # -1 -> 0, 0 -> 0.5, 1 -> 1
                    scores[collision_mask_list] = 0
                    gg.scores = scores
                    gg.widths = 0.1 * np.ones((len(gg)), dtype = np.float32)
                    grasps_geometry = gg.to_open3d_geometry_list()
                    pcd = self.loadScenePointCloud(scene_id, self.camera, ann_id)

                    o3d.visualization.draw_geometries([pcd, *grasps_geometry])
                    o3d.visualization.draw_geometries([pcd, *grasps_geometry, *model_list])
                    o3d.visualization.draw_geometries([*grasps_geometry, *model_list, t])

                if log_dict is not None:
                    log_dict[key]["scores"]["after_collision"].append(score_list.tolist())
                    log_dict[key]["confidences"]["after_collision"].append(grasp_list[:,0].tolist())

                    log_dict[key]["collision_lists"].append(collision_mask_list.tolist())
                    log_dict[key]["score_lists"].append(score_list.tolist())
                    log_dict[key]["sortidx_lists"].append(sort_idx_list)

                # sort in scene level
                grasp_confidence = grasp_list[:,0]
                indices = np.argsort(-grasp_confidence)
                grasp_list, score_list, collision_mask_list = grasp_list[indices], score_list[indices], collision_mask_list[indices]

                # score_list[collision_mask_list] = 0.1

                if do_extensive_eval:
                    TOP_K = 50

                #calculate AP
                grasp_accuracy = self.calculate_ap(TOP_K, list_coe_of_friction, score_list)

                print(key, '\tMean Accuracy for scene:%04d ann:%04d = %.3f' % (scene_id, ann_id, 100.0 * np.mean(grasp_accuracy[:,:])), flush=True)
                scene_accuracy[key].append(grasp_accuracy)
                if log_dict is not None:
                    log_dict[key]["grasp_accuracies"].append(grasp_accuracy.tolist())

                grasp_list = grasp_list[~collision_mask_list]
                score_list = score_list[~collision_mask_list]
                collision_mask_list = collision_mask_list[~collision_mask_list]
                grasp_accuracy = self.calculate_ap(TOP_K, list_coe_of_friction, score_list)
                scene_accuracy[key+'_cf'].append(grasp_accuracy)
                print(key, 'CF', '\tMean Accuracy for scene:%04d ann:%04d = %.3f' % (scene_id, ann_id, 100.0 * np.mean(grasp_accuracy[:,:])), flush=True)
            time_taken_s = time.time() - start_timestamp
            timedelta = datetime.timedelta(seconds=time_taken_s)
            print('Time taken for ann:', str(timedelta))

        return scene_accuracy


    def calculate_ap(self, TOP_K, list_coe_of_friction, score_list):
        grasp_accuracy = np.zeros((TOP_K,len(list_coe_of_friction)))
        for fric_idx, fric in enumerate(list_coe_of_friction):
            for k in range(0,TOP_K):
                if k+1 > len(score_list):
                    grasp_accuracy[k,fric_idx] = np.sum(((score_list<=fric) & (score_list>0)).astype(int))/(k+1)
                else:
                    grasp_accuracy[k,fric_idx] = np.sum(((score_list[0:k+1]<=fric) & (score_list[0:k+1]>0)).astype(int))/(k+1)
        return grasp_accuracy

    def eval_scene_and_log(self,
                           scene_id,
                           dump_folder,
                           TOP_K,
                           log_dict=None):
        log_dict = {}
        res = self.eval_scene(scene_id, dump_folder, TOP_K, log_dict=log_dict)

        with open(os.path.join(dump_folder, f'scene_{scene_id:04d}_log.json'), 'w') as f:
            json.dump(log_dict, f)

        return res

    def parallel_eval_scenes(self,
                             scene_ids,
                             dump_folder,
                             proc = 2,
                             TOP_K = None,
                             log=False):
        '''
        **Input:**

        - scene_ids: list of int of scene index.

        - dump_folder: string of the folder that saves the npy files.

        - proc: int of the number of processes to use to evaluate.

        **Output:**

        - scene_acc_list: list of the scene accuracy.
        '''
        from multiprocessing import Pool, Manager

        p = Pool(processes = proc)
        res_list = []
        for i, scene_id in enumerate(scene_ids):
            if log:
                res_list.append(p.apply_async(self.eval_scene_and_log,
                                            (scene_id, dump_folder),
                                            {'TOP_K': TOP_K}))
            else:
                res_list.append(p.apply_async(self.eval_scene,
                                            (scene_id, dump_folder), {'TOP_K': TOP_K}))
            # res_list.append(p.apply_async(self.eval_scene, (scene_id, dump_folder)))
        p.close()
        p.join()

        scene_acc_list = []
        for res in res_list:
            scene_acc_list.append(res.get())
        return scene_acc_list

    def eval_seen(self, dump_folder, proc = 2, log=False, TOP_K=50):
        '''
        **Input:**

        - dump_folder: string of the folder that saves the npy files.

        - proc: int of the number of processes to use to evaluate.

        **Output:**

        - res: numpy array of the detailed accuracy.

        - ap: float of the AP for seen split.
        '''
        res = self.parallel_eval_scenes(scene_ids = list(range(100, 130)),
                                                 dump_folder = dump_folder,
                                                 proc = proc,
                                                 log=log,
                                                 TOP_K=TOP_K)
        eval_res = {}
        aps = {}
        for key in ['unfiltered', 'collision_filtered']:
            for key2 in ['', '_cf']:
                eval_res[key + key2] = np.array([e[key + key2] for e in res])
                aps[key + key2] = np.mean(eval_res[key + key2])
                print('\n', key + key2, 'Evaluation Result:\n----------\n{}, AP Seen={}'.format(self.camera, aps[key + key2]))
        return eval_res, aps

    def eval_similar(self, dump_folder, proc = 2, log=False, TOP_K=50):
        '''
        **Input:**

        - dump_folder: string of the folder that saves the npy files.

        - proc: int of the number of processes to use to evaluate.

        **Output:**

        - res: numpy array of the detailed accuracy.

        - ap: float of the AP for similar split.
        '''
        res = self.parallel_eval_scenes(scene_ids = list(range(130, 160)),
                                                 dump_folder = dump_folder,
                                                 proc = proc,
                                                 log=log,
                                                 TOP_K=TOP_K)
        eval_res = {}
        aps = {}
        for key in ['unfiltered', 'collision_filtered']:
            for key2 in ['', '_cf']:
                eval_res[key + key2] = np.array([e[key + key2] for e in res])
                aps[key + key2] = np.mean(eval_res[key + key2])
                print('\n', key + key2, 'Evaluation Result:\n----------\n{}, AP Similar={}'.format(self.camera, aps[key + key2]))
        return res, aps

    def eval_novel(self, dump_folder, proc = 2, log=False, TOP_K=50):
        '''
        **Input:**

        - dump_folder: string of the folder that saves the npy files.

        - proc: int of the number of processes to use to evaluate.

        **Output:**

        - res: numpy array of the detailed accuracy.

        - ap: float of the AP for novel split.
        '''
        res = self.parallel_eval_scenes(scene_ids = list(range(160, 190)),
                                                 dump_folder = dump_folder,
                                                 proc = proc,
                                                 log=log,
                                                 TOP_K=TOP_K)
        eval_res = {}
        aps = {}
        for key in ['unfiltered', 'collision_filtered']:
            for key2 in ['', '_cf']:
                eval_res[key + key2] = np.array([e[key + key2] for e in res])
                aps[key + key2] = np.mean(eval_res[key + key2])
                print('\n', key + key2, 'Evaluation Result:\n----------\n{}, AP Novel={}'.format(self.camera, aps[key + key2]))
        return res, aps

    def eval_all(self, dump_folder, proc = 2, log=False, TOP_K=50):
        '''
        **Input:**

        - dump_folder: string of the folder that saves the npy files.

        - proc: int of the number of processes to use to evaluate.

        **Output:**

        - res: numpy array of the detailed accuracy.

        - ap: float of the AP for all split.
        '''
        logging.info("Evaluating on all test data")
        res = self.parallel_eval_scenes(scene_ids = list(range(100, 190)),
                                                 dump_folder = dump_folder,
                                                 proc = proc,
                                                 log=log,
                                                 TOP_K=TOP_K)
        eval_res = {}
        aps = {}
        for key in ['unfiltered', 'collision_filtered']:
            for key2 in ['', '_cf']:
                eval_res[key + key2] = np.array([e[key + key2] for e in res])
                # aps[key + key2] = np.mean(eval_res[key + key2])
                aps[key + key2] = [np.mean(eval_res[key + key2]), np.mean(eval_res[key + key2][0:30]), np.mean(eval_res[key + key2][30:60]), np.mean(eval_res[key + key2][60:90])]
                # ap = [np.mean(res), np.mean(res[0:30]), np.mean(res[30:60]), np.mean(res[60:90])]
                logging.info('\nEvaluation Result:\n----------\n{}, AP={}, AP Seen={}, AP Similar={}, AP Novel={}'.format(self.camera, aps[key + key2][0], aps[key + key2][1], aps[key + key2][2], aps[key + key2][3]))
        return eval_res, aps

    def eval_train(self, dump_folder, proc = 2, log=False, TOP_K=50):
        '''
        **Input:**

        - dump_folder: string of the folder that saves the npy files.

        - proc: int of the number of processes to use to evaluate.

        **Output:**

        - res: numpy array of the detailed accuracy.

        - ap: float of the AP for all split.
        '''
        res = self.parallel_eval_scenes(scene_ids = list(range(0, 100)),
                                                 dump_folder = dump_folder,
                                                 proc = proc,
                                                 log=log,
                                                 TOP_K=TOP_K)
        eval_res = {}
        aps = {}
        for key in ['unfiltered', 'collision_filtered']:
            for key2 in ['', '_cf']:
                eval_res[key + key2] = np.array([e[key + key2] for e in res])
                aps[key + key2] = np.mean(eval_res[key + key2])
                print('\n', key + key2, 'Evaluation Result:\n----------\n{}, AP Train={}'.format(self.camera, aps[key + key2]))
        return res, aps
