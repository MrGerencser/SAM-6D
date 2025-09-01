import os
import sys
import json
import numpy as np
import torch
import logging
from PIL import Image
import glob
from omegaconf import DictConfig, OmegaConf
from hydra import initialize, compose
from hydra.utils import instantiate

# Add paths 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from utils.poses.pose_utils import get_obj_poses_from_template_level, load_index_level_in_level2
from utils.bbox_utils import CropResizePad
from model.utils import Detections, convert_npz_to_json
from utils.inout import load_json, save_json_bop23

class InstanceSegmentationModel:
    def __init__(self, segmentor_model="sam", stability_score_thresh=0.97):
        """Initialize the instance segmentation model once - EXACTLY like run_inference_custom.py"""
        self.segmentor_model_name = segmentor_model
        self.stability_score_thresh = stability_score_thresh
        
        # Set working directory to ISM directory
        original_cwd = os.getcwd()
        os.chdir(BASE_DIR)
        
        try:
            # Load config - EXACTLY like run_inference_custom.py
            with initialize(version_base=None, config_path="configs"):
                cfg = compose(config_name='run_inference.yaml')

            # Configure segmentor model - EXACTLY like run_inference_custom.py
            if segmentor_model == "sam":
                with initialize(version_base=None, config_path="configs/model"):
                    cfg.model = compose(config_name='ISM_sam.yaml')
                cfg.model.segmentor_model.stability_score_thresh = stability_score_thresh
            elif segmentor_model == "fastsam":
                with initialize(version_base=None, config_path="configs/model"):
                    cfg.model = compose(config_name='ISM_fastsam.yaml')
            elif segmentor_model == "sam2":
                with initialize(version_base=None, config_path="configs/model"):
                    cfg.model = compose(config_name='ISM_sam2.yaml')
            else:
                raise ValueError(f"The segmentor_model {segmentor_model} is not supported!")

            # Initialize model - EXACTLY like run_inference_custom.py
            logging.info("Initializing model")
            self.model = instantiate(cfg.model)
            
            # Setup device and move models - EXACTLY like run_inference_custom.py
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.descriptor_model.model = self.model.descriptor_model.model.to(self.device)
            self.model.descriptor_model.model.device = self.device
            
            # Move segmentor model to device - EXACTLY like run_inference_custom.py
            if hasattr(self.model.segmentor_model, "predictor"):
                self.model.segmentor_model.predictor.model = (
                    self.model.segmentor_model.predictor.model.to(self.device)
                )
            elif hasattr(self.model.segmentor_model, "model"):
                # For FastSAM
                if hasattr(self.model.segmentor_model.model, "setup_model"):
                    self.model.segmentor_model.model.setup_model(device=self.device, verbose=True)
            # SAM2 has no predictor/model setup needed here
            
            logging.info(f"Moving models to {self.device} done!")
            
            # Template cache
            self.template_cache = {}
            self.processing_config = OmegaConf.create({"image_size": 224})
            self.proposal_processor = CropResizePad(self.processing_config.image_size)
            
            logging.info("Instance Segmentation Model loaded successfully")
            
        finally:
            os.chdir(original_cwd)
    
    def load_templates_and_setup_ref_data(self, template_dir, cad_path):
        """Load templates and setup ref_data - EXACTLY like run_inference_custom.py"""
        cache_key = f"{template_dir}_{cad_path}"
        if cache_key in self.template_cache:
            logging.info(f"Using cached templates for {cache_key}")
            self.model.ref_data = self.template_cache[cache_key]
            return
        
        logging.info("Initializing template")
        
        # Change to ISM directory
        original_cwd = os.getcwd()
        os.chdir(BASE_DIR)
        
        try:
            # Load templates - EXACTLY like run_inference_custom.py
            num_templates = len(glob.glob(f"{template_dir}/*.npy"))
            boxes, masks, templates = [], [], []
            
            for idx in range(num_templates):
                image = Image.open(os.path.join(template_dir, f'rgb_{idx}.png'))
                mask = Image.open(os.path.join(template_dir, f'mask_{idx}.png'))
                boxes.append(mask.getbbox())

                image = torch.from_numpy(np.array(image.convert("RGB")) / 255).float()
                mask = torch.from_numpy(np.array(mask.convert("L")) / 255).float()
                image = image * mask[:, :, None]
                templates.append(image)
                masks.append(mask.unsqueeze(-1))
            
            templates = torch.stack(templates).permute(0, 3, 1, 2)
            masks = torch.stack(masks).permute(0, 3, 1, 2)
            boxes = torch.tensor(np.array(boxes))
            
            # Process templates - EXACTLY like run_inference_custom.py
            templates = self.proposal_processor(images=templates, boxes=boxes).to(self.device)
            masks_cropped = self.proposal_processor(images=masks, boxes=boxes).to(self.device)

            # Setup ref_data - EXACTLY like run_inference_custom.py
            self.model.ref_data = {}
            self.model.ref_data["descriptors"] = self.model.descriptor_model.compute_features(
                            templates, token_name="x_norm_clstoken"
                        ).unsqueeze(0).data
            self.model.ref_data["appe_descriptors"] = self.model.descriptor_model.compute_masked_patch_feature(
                            templates, masks_cropped[:, 0, :, :]
                        ).unsqueeze(0).data

            # Setup pose data - EXACTLY like run_inference_custom.py
            template_poses = get_obj_poses_from_template_level(level=2, pose_distribution="all")
            template_poses[:, :3, 3] *= 0.4
            poses = torch.tensor(template_poses).to(torch.float32).to(self.device)
            poses_filtered = poses[load_index_level_in_level2(0, "all"), :, :]
            
            # Load mesh and sample points - EXACTLY like run_inference_custom.py
            import trimesh
            mesh = trimesh.load_mesh(cad_path)
            model_points = mesh.sample(2048).astype(np.float32) / 1000.0
            pointcloud = torch.tensor(model_points).unsqueeze(0).data.to(self.device)
            
            # Add to ref_data - EXACTLY like run_inference_custom.py
            self.model.ref_data["poses"] = poses_filtered
            self.model.ref_data["pointcloud"] = pointcloud
            
            # Cache the ref_data
            self.template_cache[cache_key] = self.model.ref_data.copy()
            
            logging.info(f"Templates loaded and cached for {cache_key}")
            
        finally:
            os.chdir(original_cwd)
    
    def infer_segmentation(self, rgb_path, depth_path, cam_path, cad_path, template_dir, output_dir=None):
        """Run inference - EXACTLY like run_inference_custom.py"""
        
        # Change to ISM directory
        original_cwd = os.getcwd()
        os.chdir(BASE_DIR)
        
        try:
            # Load templates (uses cache if available)
            self.load_templates_and_setup_ref_data(template_dir, cad_path)
            
            # Load RGB image - EXACTLY like run_inference_custom.py
            rgb = Image.open(rgb_path).convert("RGB")
            rgb_np = np.array(rgb)
            
            # Run segmentation - EXACTLY like run_inference_custom.py
            detections_dict = self.model.segmentor_model.generate_masks(rgb_np)
            detections = Detections(detections_dict)
            
            if len(detections.masks) == 0:
                logging.warning("No detections found")
                return []
            
            # *** ADD THE COMPLETE PIPELINE FROM run_inference_custom.py ***
            
            # Compute descriptors - EXACTLY like run_inference_custom.py
            query_descriptors, query_appe_descriptors = self.model.descriptor_model.forward(rgb_np, detections)

            # Matching descriptors - EXACTLY like run_inference_custom.py
            (
                idx_selected_proposals,
                pred_idx_objects,
                semantic_score,
                best_template,
            ) = self.model.compute_semantic_score(query_descriptors)

            # Update detections - EXACTLY like run_inference_custom.py
            detections.filter(idx_selected_proposals)
            query_appe_descriptors = query_appe_descriptors[idx_selected_proposals, :]

            # Compute the appearance score - EXACTLY like run_inference_custom.py
            appe_scores, ref_aux_descriptor = self.model.compute_appearance_score(best_template, pred_idx_objects, query_appe_descriptors)

            # Compute the geometric score - EXACTLY like run_inference_custom.py
            batch = self._batch_input_data(depth_path, cam_path)
            image_uv = self.model.project_template_to_image(best_template, pred_idx_objects, batch, detections.masks)

            geometric_score, visible_ratio = self.model.compute_geometric_score(
                image_uv, detections, query_appe_descriptors, ref_aux_descriptor, visible_thred=self.model.visible_thred
            )

            # Final score - EXACTLY like run_inference_custom.py
            final_score = (semantic_score + appe_scores + geometric_score*visible_ratio) / (1 + 1 + visible_ratio)

            # Add required attributes - EXACTLY like run_inference_custom.py
            detections.add_attribute("scores", final_score)
            detections.add_attribute("object_ids", torch.zeros_like(final_score))   
            
            # Convert to numpy - EXACTLY like run_inference_custom.py
            detections.to_numpy()
            
            # Save results if output_dir provided - EXACTLY like run_inference_custom.py
            if output_dir:
                results_dir = os.path.join(output_dir, 'sam6d_results')
                os.makedirs(results_dir, exist_ok=True)
                
                save_path = os.path.join(results_dir, "detection_ism")
                detections.save_to_file(0, 0, 0, save_path, "Custom", return_results=False)
                detections_json = convert_npz_to_json(idx=0, list_npz_paths=[save_path + ".npz"])
                save_json_bop23(save_path + ".json", detections_json)
                
                return detections_json
            
            # Convert to simple format for return
            results = []
            for i in range(len(detections.masks)):
                score = float(detections.scores[i]) if hasattr(detections, 'scores') else 1.0
                
                bbox = detections.boxes[i] if hasattr(detections, 'boxes') else []
                if hasattr(bbox, 'tolist'):
                    bbox = bbox.tolist()
                
                mask = detections.masks[i]
                
                result = {
                    'score': score,
                    'segmentation': mask,
                    'bbox': bbox,
                    'category_id': 1
                }
                results.append(result)
            
            return results
            
        finally:
            os.chdir(original_cwd)

    def _batch_input_data(self, depth_path, cam_path):
        """Prepare batch input data - EXACTLY like run_inference_custom.py"""
        import imageio.v2 as imageio
        
        batch = {}
        cam_info = load_json(cam_path)
        depth = np.array(imageio.imread(depth_path)).astype(np.int32)
        cam_K = np.array(cam_info['cam_K']).reshape((3, 3))
        depth_scale = np.array(cam_info['depth_scale'])

        batch["depth"] = torch.from_numpy(depth).unsqueeze(0).to(self.device)
        batch["cam_intrinsic"] = torch.from_numpy(cam_K).unsqueeze(0).to(self.device)
        batch['depth_scale'] = torch.from_numpy(depth_scale).unsqueeze(0).to(self.device)
        return batch

# Global model instance (singleton pattern)
_ism_model = None

def get_ism_model(segmentor_model="sam", stability_score_thresh=0.97):
    """Get the global instance segmentation model instance"""
    global _ism_model
    if _ism_model is None:
        logging.info("Loading ISM model for the first time...")
        _ism_model = InstanceSegmentationModel(segmentor_model, stability_score_thresh)
    return _ism_model

def infer_segmentation_batch(rgb_path, depth_path, cam_path, cad_path, template_dir, 
                           segmentor_model="sam", stability_score_thresh=0.97, output_dir=None):
    """Convenience function for batch inference"""
    model = get_ism_model(segmentor_model, stability_score_thresh)
    return model.infer_segmentation(rgb_path, depth_path, cam_path, cad_path, template_dir, output_dir)