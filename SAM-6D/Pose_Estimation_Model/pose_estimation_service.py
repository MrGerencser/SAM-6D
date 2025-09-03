import os
import sys
import json
import numpy as np
import torch
import cv2
import importlib
from PIL import Image
import torchvision.transforms as transforms
import trimesh
import pycocotools.mask as cocomask

# Add paths correctly
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'provider'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
sys.path.append(os.path.join(BASE_DIR, 'model'))
sys.path.append(os.path.join(BASE_DIR, 'model', 'pointnet2'))

from data_utils import (
    load_im,
    get_bbox,
    get_point_cloud_from_depth,
    get_resize_rgb_choose,
)
from draw_utils import draw_detections
import gorilla

class PoseEstimationModel:
    def __init__(self, config_path="config/base.yaml", checkpoint_path=None, gpu_id="0"):
        """Initialize the pose estimation model once"""
        
        # Set working directory to pose estimation model directory
        original_cwd = os.getcwd()
        os.chdir(BASE_DIR)
        
        try:
            # Load config using gorilla
            self.cfg = gorilla.Config.fromfile(config_path)
            gorilla.utils.set_cuda_visible_devices(gpu_ids=gpu_id)
            
            # Load model
            MODEL = importlib.import_module("pose_estimation_model")
            self.model = MODEL.Net(self.cfg.model)
            self.model = self.model.cuda()
            self.model.eval()
            
            # Load checkpoint
            if checkpoint_path is None:
                checkpoint_path = os.path.join(BASE_DIR, 'checkpoints', 'sam-6d-pem-base.pth')
            gorilla.solver.load_checkpoint(model=self.model, filename=checkpoint_path)
            
            # Setup transforms
            self.rgb_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            # Template cache
            self.template_cache = {}
            
            print("Pose estimation model loaded successfully")
            
        finally:
            # Restore original working directory
            os.chdir(original_cwd)
    
    def load_templates(self, template_path, cad_path):
        """Load and cache templates for a specific model"""
        cache_key = template_path
        if cache_key in self.template_cache:
            print(f"Using cached templates for {cache_key}")
            return self.template_cache[cache_key]
        
        print(f"Loading templates from {template_path}...")
        
        # Change to pose estimation directory for template loading
        original_cwd = os.getcwd()
        os.chdir(BASE_DIR)
        
        try:
            all_tem, all_tem_pts, all_tem_choose = self._get_templates(template_path)
            
            with torch.no_grad():
                all_tem_pts, all_tem_feat = self.model.feature_extraction.get_obj_feats(
                    all_tem, all_tem_pts, all_tem_choose
                )
            
            # Cache the processed templates
            self.template_cache[cache_key] = {
                'tem_pts': all_tem_pts,
                'tem_feat': all_tem_feat
            }
            
            print(f"Templates loaded and cached for {cache_key}")
            return self.template_cache[cache_key]
            
        finally:
            os.chdir(original_cwd)
    
    def infer_pose(self, rgb_path, depth_path, cam_path, cad_path, seg_path, 
                   template_path, det_score_thresh=0.2, output_dir=None, debug_vis=True):
        """Run pose inference on given data"""
        
        # Load templates (uses cache if available)
        templates = self.load_templates(template_path, cad_path)
        
        # Change to pose estimation directory for inference
        original_cwd = os.getcwd()
        os.chdir(BASE_DIR)
        
        try:
            # Prepare input data
            input_data, img, whole_pts, model_points, detections = self._get_test_data(
                rgb_path, depth_path, cam_path, cad_path, seg_path, det_score_thresh
            )
            
            if input_data['pts'].size(0) == 0:
                print("No valid detections found")
                return []
            
            ninstance = input_data['pts'].size(0)
            
            # Run inference
            with torch.no_grad():
                input_data['dense_po'] = templates['tem_pts'].repeat(ninstance, 1, 1)
                input_data['dense_fo'] = templates['tem_feat'].repeat(ninstance, 1, 1)
                out = self.model(input_data)
            
            # Process results
            if 'pred_pose_score' in out.keys():
                pose_scores = out['pred_pose_score'] * out['score']
            else:
                pose_scores = out['score']
                
            pose_scores = pose_scores.detach().cpu().numpy()
            pred_rot = out['pred_R'].detach().cpu().numpy()
            pred_trans = out['pred_t'].detach().cpu().numpy() * 1000  # Convert to mm
            
            # Format results
            results = []
            for idx, det in enumerate(detections):
                result = {
                    'score': float(pose_scores[idx]),
                    'R': pred_rot[idx].tolist(),
                    't': pred_trans[idx].tolist(),
                    'segmentation': det['segmentation'],
                    'bbox': det.get('bbox', []),
                    'category_id': det.get('category_id', 1)
                }
                results.append(result)
            
            # CREATE VISUALIZATION
            if output_dir:
                results_dir = os.path.join(output_dir, 'sam6d_results')
                os.makedirs(results_dir, exist_ok=True)
                
                # Save results
                with open(os.path.join(results_dir, 'detection_pem.json'), 'w') as f:
                    json.dump(results, f)

                if debug_vis:
                    save_path = os.path.join(results_dir, 'vis_pem.png')
                    valid_masks = pose_scores == pose_scores.max()
                    K_vis = input_data['K'].detach().cpu().numpy()[valid_masks]
                    
                    print(f"DEBUG: Creating visualization at {save_path}")
                    print(f"DEBUG: valid_masks shape: {valid_masks.shape}, sum: {np.sum(valid_masks)}")
                    print(f"DEBUG: K_vis shape: {K_vis.shape}")
                    print(f"DEBUG: img shape: {img.shape}")
                    print(f"DEBUG: pred_rot[valid_masks] shape: {pred_rot[valid_masks].shape}")
                    print(f"DEBUG: pred_trans[valid_masks] shape: {pred_trans[valid_masks].shape}")
                    
                    vis_img = self.visualize(img, pred_rot[valid_masks], pred_trans[valid_masks], model_points*1000, K_vis, save_path)
                    vis_img.save(save_path)
                    print(f"✅ Pose visualization saved to {save_path}")
                    

            return results
            
        finally:
            os.chdir(original_cwd)
    
    def visualize(self, rgb, pred_rot, pred_trans, model_points, K, save_path):
        """Visualize pose estimation results - EXACTLY like run_inference_custom.py"""
        from draw_utils import draw_detections
        
        img = draw_detections(rgb, pred_rot, pred_trans, model_points, K, color=(255, 0, 0))
        img = Image.fromarray(np.uint8(img))
        img.save(save_path)
        prediction = Image.open(save_path)
        
        # concat side by side in PIL
        rgb = Image.fromarray(np.uint8(rgb))
        img = np.array(img)
        concat = Image.new('RGB', (img.shape[1] + prediction.size[0], img.shape[0]))
        concat.paste(rgb, (0, 0))
        concat.paste(prediction, (img.shape[1], 0))
        return concat
    
    def _get_templates(self, path):
        """Load templates from path"""
        n_template_view = self.cfg.test_dataset.n_template_view
        all_tem = []
        all_tem_choose = []
        all_tem_pts = []

        total_nView = 42
        for v in range(n_template_view):
            i = int(total_nView / n_template_view * v)
            tem, tem_choose, tem_pts = self._get_template(path, i)
            all_tem.append(torch.FloatTensor(tem).unsqueeze(0).cuda())
            all_tem_choose.append(torch.IntTensor(tem_choose).long().unsqueeze(0).cuda())
            all_tem_pts.append(torch.FloatTensor(tem_pts).unsqueeze(0).cuda())
        return all_tem, all_tem_pts, all_tem_choose
    
    def _get_template(self, path, tem_index):
        """Load a single template"""
        rgb_path = os.path.join(path, f'rgb_{tem_index}.png')
        mask_path = os.path.join(path, f'mask_{tem_index}.png')
        xyz_path = os.path.join(path, f'xyz_{tem_index}.npy')

        rgb = load_im(rgb_path).astype(np.uint8)
        xyz = np.load(xyz_path).astype(np.float32) / 1000.0  
        mask = load_im(mask_path).astype(np.uint8) == 255

        bbox = get_bbox(mask)
        y1, y2, x1, x2 = bbox
        mask = mask[y1:y2, x1:x2]

        rgb = rgb[:,:,::-1][y1:y2, x1:x2, :]
        if self.cfg.test_dataset.rgb_mask_flag:
            rgb = rgb * (mask[:,:,None]>0).astype(np.uint8)

        rgb = cv2.resize(rgb, (self.cfg.test_dataset.img_size, self.cfg.test_dataset.img_size), 
                        interpolation=cv2.INTER_LINEAR)
        rgb = self.rgb_transform(np.array(rgb))

        choose = (mask>0).astype(np.float32).flatten().nonzero()[0]
        if len(choose) <= self.cfg.test_dataset.n_sample_template_point:
            choose_idx = np.random.choice(np.arange(len(choose)), 
                                        self.cfg.test_dataset.n_sample_template_point)
        else:
            choose_idx = np.random.choice(np.arange(len(choose)), 
                                        self.cfg.test_dataset.n_sample_template_point, replace=False)
        choose = choose[choose_idx]
        xyz = xyz[y1:y2, x1:x2, :].reshape((-1, 3))[choose, :]

        rgb_choose = get_resize_rgb_choose(choose, [y1, y2, x1, x2], self.cfg.test_dataset.img_size)
        return rgb, rgb_choose, xyz
    
    def _get_test_data(self, rgb_path, depth_path, cam_path, cad_path, seg_path, det_score_thresh):
        """Prepare test data from input files - EXACT COPY from working run_inference_custom.py"""
        dets = []
        with open(seg_path) as f:
            dets_ = json.load(f) # keys: scene_id, image_id, category_id, bbox, score, segmentation
        for det in dets_:
            if det['score'] > det_score_thresh:
                dets.append(det)
        del dets_

        cam_info = json.load(open(cam_path))
        K = np.array(cam_info['cam_K']).reshape(3, 3)

        whole_image = load_im(rgb_path).astype(np.uint8)
        if len(whole_image.shape)==2:
            whole_image = np.concatenate([whole_image[:,:,None], whole_image[:,:,None], whole_image[:,:,None]], axis=2)
        whole_depth = load_im(depth_path).astype(np.float32) * cam_info['depth_scale'] / 1000.0
        whole_pts = get_point_cloud_from_depth(whole_depth, K)

        mesh = trimesh.load_mesh(cad_path)
        model_points = mesh.sample(self.cfg.test_dataset.n_sample_model_point).astype(np.float32) / 1000.0
        radius = np.max(np.linalg.norm(model_points, axis=1))

        all_rgb = []
        all_cloud = []
        all_rgb_choose = []
        all_score = []
        all_dets = []
        
        for inst in dets:
            seg = inst['segmentation']
            score = inst['score']

            # mask - EXACT SAME AS ORIGINAL
            h, w = seg['size']
            try:
                rle = cocomask.frPyObjects(seg, h, w)
            except:
                rle = seg
            mask = cocomask.decode(rle)
            mask = np.logical_and(mask > 0, whole_depth > 0)
            if np.sum(mask) > 32:
                bbox = get_bbox(mask)
                y1, y2, x1, x2 = bbox
            else:
                continue
            mask = mask[y1:y2, x1:x2]
            choose = mask.astype(np.float32).flatten().nonzero()[0]

            # pts
            cloud = whole_pts.copy()[y1:y2, x1:x2, :].reshape(-1, 3)[choose, :]
            center = np.mean(cloud, axis=0)
            tmp_cloud = cloud - center[None, :]
            flag = np.linalg.norm(tmp_cloud, axis=1) < radius * 1.2
            if np.sum(flag) < 4:
                continue
            choose = choose[flag]
            cloud = cloud[flag]

            if len(choose) <= self.cfg.test_dataset.n_sample_observed_point:
                choose_idx = np.random.choice(np.arange(len(choose)), 
                                            self.cfg.test_dataset.n_sample_observed_point)
            else:
                choose_idx = np.random.choice(np.arange(len(choose)), 
                                            self.cfg.test_dataset.n_sample_observed_point, replace=False)
            choose = choose[choose_idx]
            cloud = cloud[choose_idx]

            # rgb
            rgb = whole_image.copy()[y1:y2, x1:x2, :][:,:,::-1]
            if self.cfg.test_dataset.rgb_mask_flag:
                rgb = rgb * (mask[:,:,None]>0).astype(np.uint8)
            rgb = cv2.resize(rgb, (self.cfg.test_dataset.img_size, self.cfg.test_dataset.img_size), 
                        interpolation=cv2.INTER_LINEAR)
            rgb = self.rgb_transform(np.array(rgb))
            rgb_choose = get_resize_rgb_choose(choose, [y1, y2, x1, x2], self.cfg.test_dataset.img_size)

            all_rgb.append(torch.FloatTensor(rgb))
            all_cloud.append(torch.FloatTensor(cloud))
            all_rgb_choose.append(torch.IntTensor(rgb_choose).long())
            all_score.append(score)
            all_dets.append(inst)

        # Handle empty case
        if len(all_rgb) == 0:
            ret_dict = {
                'pts': torch.empty(0, 0, 3).cuda(),
                'rgb': torch.empty(0, 3, 0, 0).cuda(),
                'rgb_choose': torch.empty(0, 0).cuda(),
                'score': torch.empty(0).cuda(),
                'model': torch.empty(0, 0, 3).cuda(),
                'K': torch.empty(0, 3, 3).cuda()
            }
            return ret_dict, whole_image, whole_pts.reshape(-1, 3), model_points, []

        ret_dict = {}
        ret_dict['pts'] = torch.stack(all_cloud).cuda()
        ret_dict['rgb'] = torch.stack(all_rgb).cuda()
        ret_dict['rgb_choose'] = torch.stack(all_rgb_choose).cuda()
        ret_dict['score'] = torch.FloatTensor(all_score).cuda()

        ninstance = ret_dict['pts'].size(0)
        ret_dict['model'] = torch.FloatTensor(model_points).unsqueeze(0).repeat(ninstance, 1, 1).cuda()
        ret_dict['K'] = torch.FloatTensor(K).unsqueeze(0).repeat(ninstance, 1, 1).cuda()
        return ret_dict, whole_image, whole_pts.reshape(-1, 3), model_points, all_dets

# Global model instance (TRUE singleton pattern)
_pose_model = None

def get_pose_model(config_path="config/base.yaml", checkpoint_path=None, gpu_id="0"):
    """Get the global pose estimation model instance - loads ONLY ONCE"""
    global _pose_model
    if _pose_model is None:
        print("Loading pose estimation model for the first time...")
        _pose_model = PoseEstimationModel(config_path, checkpoint_path, gpu_id)
    return _pose_model

def infer_pose_batch(rgb_path, depth_path, cam_path, cad_path, seg_path, 
                    template_path, det_score_thresh=0.2):
    """Convenience function for batch inference"""
    model = get_pose_model()
    return model.infer_pose(rgb_path, depth_path, cam_path, cad_path, seg_path, 
                          template_path, det_score_thresh)