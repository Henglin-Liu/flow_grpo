import os
import sys
import cv2
import numpy as np
import torch
from huggingface_hub import snapshot_download
from insightface.app import FaceAnalysis
from insightface.utils import face_align
from PIL import Image
from torchvision import models, transforms
from typing import List, Union, Optional
from transformers import CLIPModel, CLIPProcessor
import matplotlib.pyplot as plt

class FaceSimilarityFIDScorer:
    def __init__(
        self,
        device: str = "cuda",
        model_path: str = "/m2v_intern/liuhenglin/code/video_gen/flow_grpo/ckpts",
        clip_model_path: str = "/m2v_intern/liuhenglin/code/video_gen/flow_grpo/ckpts/data_process/clip-vit-base-patch32",
        calculate_cur: bool = True,
        calculate_arc: bool = True,
        calculate_fid: bool = False,
        calculate_clip: bool = False
    ):
        """
        Face similarity, FID, and CLIP score calculator with configurable metrics
        
        :param device: Device to run calculations on ('cuda' or 'cpu')
        :param model_path: Path to store/download face models
        :param clip_model_path: Path to store/download CLIP model
        :param calculate_cur: Whether to calculate CurricularFace similarity
        :param calculate_arc: Whether to calculate ArcFace similarity
        :param calculate_fid: Whether to calculate FID score
        :param calculate_clip: Whether to calculate CLIP score
        """
        self.device = device
        self.model_path = model_path
        self.clip_model_path = clip_model_path
        self.calculate_cur = calculate_cur
        self.calculate_arc = calculate_arc
        self.calculate_fid = calculate_fid
        self.calculate_clip = calculate_clip
        
        # Initialize paths and ensure they exist
        self._setup_paths()
        
        # Initialize models based on what we need to calculate
        if self.calculate_arc:
            self.face_arc_model = self._init_face_arc_model()
        
        if self.calculate_cur:
            self.face_cur_model = self._init_face_cur_model()
        
        if self.calculate_fid:
            self.fid_model = self._init_fid_model()
        
        if self.calculate_clip:
            self.clip_model, self.clip_processor = self._init_clip_model()
        else:
            self.clip_model = None
            self.clip_processor = None
        
        # Initialize transforms
        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor()
        ])

    def _setup_paths(self):
        """Ensure model paths exist and download if needed"""
        # Face models
        if not os.path.exists(self.model_path):
            print("Face model not found, downloading from Hugging Face...")
            snapshot_download(repo_id="BestWishYsh/ConsisID-preview", local_dir=self.model_path)
        else:
            print(f"Face model already exists in {self.model_path}, skipping download.")
            
        self.face_arc_path = os.path.join(self.model_path, "face_encoder")
        self.face_cur_path = os.path.join(self.face_arc_path, "glint360k_curricular_face_r101_backbone.bin")
        
        # CLIP model
        if not os.path.exists(self.clip_model_path):
            print("CLIP model not found, downloading from Hugging Face...")
            snapshot_download(repo_id="openai/clip-vit-base-patch32", local_dir=self.clip_model_path)
        else:
            print(f"CLIP model already exists in {self.clip_model_path}, skipping download.")

    def _init_face_arc_model(self):
        """Initialize ArcFace model"""
        model = FaceAnalysis(root=self.face_arc_path, providers=['CUDAExecutionProvider'])
        model.prepare(ctx_id=0, det_size=(320, 320))
        return model

    def _init_face_cur_model(self):
        """Initialize CurricularFace model"""
        from flow_grpo.on_going_module.curricularface import get_model
        model = get_model('IR_101')([112, 112])
        model.load_state_dict(torch.load(self.face_cur_path, map_location="cpu"))
        model = model.to(self.device)
        model.eval()
        return model

    def _init_fid_model(self):
        """Initialize FID model (InceptionV3)"""
        model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
        model.fc = torch.nn.Identity()  # Remove final classification layer
        model.eval()
        model = model.to(self.device)
        return model

    def _init_clip_model(self):
        """Initialize CLIP model and processor"""
        if not self.calculate_clip:
            return None, None
            
        model = CLIPModel.from_pretrained(self.clip_model_path)
        processor = CLIPProcessor.from_pretrained(self.clip_model_path)
        model = model.to(self.device)
        return model, processor

    def _process_image(self, image: np.ndarray):
        """Process image to get aligned face and embeddings"""
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Get face keypoints
        face_info = self._get_face_keypoints(image_bgr)
        if face_info is None:
            padded_image, sub_coord = self._pad_np_bgr_image(image_bgr)
            face_info = self._get_face_keypoints(padded_image)
            if face_info is None:
                raise ValueError("No face detected in the image")
            face_kps = face_info['kps'] - np.array(sub_coord)
        else:
            face_kps = face_info['kps']
            
        arcface_embedding = face_info['embedding']
        norm_face = face_align.norm_crop(image_bgr, landmark=face_kps, image_size=224)
        align_face = cv2.cvtColor(norm_face, cv2.COLOR_BGR2RGB)
        
        return align_face, arcface_embedding

    def _get_face_keypoints(self, image_bgr):
        """Get face keypoints from image"""
        face_info = self.face_arc_model.get(image_bgr)
        if len(face_info) > 0:
            return sorted(face_info, key=lambda x: (x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1]
        return None

    def _pad_np_bgr_image(self, np_image, scale=1.25):
        """Pad image for better face detection"""
        assert scale >= 1.0, "scale should be >= 1.0"
        pad_scale = scale - 1.0
        h, w = np_image.shape[:2]
        top = bottom = int(h * pad_scale)
        left = right = int(w * pad_scale)
        return cv2.copyMakeBorder(np_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(128, 128, 128)), (left, top)

    def _inference(self, img: np.ndarray):
        """Get face embedding from aligned face"""
        img = cv2.resize(img, (112, 112))
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        img.div_(255).sub_(0.5).div_(0.5)
        embedding = self.face_cur_model(img).detach().cpu().numpy()[0]
        return embedding / np.linalg.norm(embedding)


    def _batch_cosine_similarity(self, embedding1, embedding2):
        """Calculate cosine similarity between embeddings"""
        embedding1 = torch.tensor(embedding1).to(self.device)
        embedding2 = torch.tensor(embedding2).to(self.device)
        return torch.nn.functional.cosine_similarity(embedding1, embedding2, dim=-1).cpu().numpy()

    def _matrix_sqrt(self, matrix):
        """Calculate matrix square root for FID"""
        eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
        sqrt_eigenvalues = torch.sqrt(torch.clamp(eigenvalues, min=0))
        sqrt_matrix = (eigenvectors * sqrt_eigenvalues).mm(eigenvectors.T)
        return sqrt_matrix

    def _calculate_fid(self, real_activations, fake_activations):
        """Calculate FID score"""
        real_activations_tensor = torch.tensor(real_activations).to(self.device)
        fake_activations_tensor = torch.tensor(fake_activations).to(self.device)

        mu1 = real_activations_tensor.mean(dim=0)
        sigma1 = torch.cov(real_activations_tensor.T)
        mu2 = fake_activations_tensor.mean(dim=0)
        sigma2 = torch.cov(fake_activations_tensor.T)

        ssdiff = torch.sum((mu1 - mu2) ** 2)
        covmean = self._matrix_sqrt(sigma1.mm(sigma2))
        if torch.is_complex(covmean):
            covmean = covmean.real
        fid = ssdiff + torch.trace(sigma1 + sigma2 - 2 * covmean)
        return fid.item()

    def _get_activations(self, images, batch_size=16):
        """Get activations from FID model"""
        self.fid_model.eval()
        activations = []
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch = images[i:i + batch_size]
                pred = self.fid_model(batch)
                activations.append(pred)
        activations = torch.cat(activations, dim=0).cpu().numpy()
        if activations.shape[0] == 1:
            activations = np.repeat(activations, 2, axis=0)
        return activations

    def _compute_clip_score(self, frames: List[np.ndarray], prompt: str) -> float:
        """Compute CLIP score for given frames and prompt"""
        if self.clip_model is None or self.clip_processor is None:
            return 0.0
            
        inputs = self.clip_processor(
            text=prompt,
            images=frames,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        inputs.to(self.device)
        outputs = self.clip_model(**inputs)
        logits_per_image = outputs.logits_per_image
        return logits_per_image.mean().item()
    
    def _load_image(self, image: Union[str, Image.Image, np.ndarray]) -> np.ndarray:
        """Load and convert input image to numpy array"""
        if isinstance(image, str):
            return np.array(Image.open(image).convert("RGB"))
        elif isinstance(image, Image.Image):
            return np.array(image.convert("RGB"))
        elif isinstance(image, np.ndarray):
            return image
        else:
            raise TypeError("Input must be path string, PIL Image, or numpy array")
    
    def __call__(
            self,
            video_frames_list: Union[List[np.ndarray], np.ndarray],
            image_list: Union[List[np.ndarray], np.ndarray, List[str]],
            clip_prompt: Optional[List[str]] = None,
            reduction_mean: bool = True
        ) -> List[dict]:
        """
        Calculate specified metrics between videos and reference images
        
        :param video_frames_list: [[(H,W,C)]*frames_num]
        :param np_image_list: [(H,W,C)]
        :param clip_prompt: [str]
        :return: List of dictionaries containing requested scores for each pair
        """
        
        # Validate input lengths
        if len(video_frames_list) != len(image_list):
            raise ValueError("Number of video paths must match number of image paths")
        if clip_prompt is not None and len(clip_prompt) != len(video_frames_list):
            raise ValueError("Number of clip prompts must match number of video paths")
        
        all_results = []
        
        for i, (video_frames, image) in enumerate(zip(video_frames_list, image_list)):
            # Get current clip prompt if available
            current_clip_prompt = clip_prompt[i] if clip_prompt is not None else None
            
            # Process reference image
            if isinstance(image, str):
                np_image = self._load_image(image)
            else:
                np_image = image
            
            # Initialize results dictionary for this pair
            results = {}
            
            # Initialize face-related variables
            align_face_image = None
            arcface_image_embedding = None
            cur_image_embedding = None
            real_activations = None
            
            # Only process face if we need face-related metrics
            if self.calculate_arc or self.calculate_cur or self.calculate_fid:
                try:
                    align_face_image, arcface_image_embedding = self._process_image(np_image)
                    
                    if self.calculate_cur:
                        cur_image_embedding = self._inference(align_face_image)
                    
                    if self.calculate_fid:
                        align_face_image_pil = Image.fromarray(align_face_image)
                        real_image = self.transform(align_face_image_pil).unsqueeze(0).to(self.device)
                        real_activations = self._get_activations(real_image)
                except Exception as e:
                    print(f"Failed to process reference image face: {str(e)}, face path is {image}")
                    # Set face-related scores to 0 if face detection fails
                    if self.calculate_arc:
                        results['arc_score'] = 0.0
                    if self.calculate_cur:
                        results['cur_score'] = 0.0
                    if self.calculate_fid:
                        results['fid_score'] = 0.0
            
            # Process video frames
            # Initialize score trackers
            if self.calculate_cur and 'cur_score' not in results:
                cur_scores = []
            if self.calculate_arc and 'arc_score' not in results:
                arc_scores = []
            if self.calculate_fid and 'fid_score' not in results:
                fid_scores = []
            if self.calculate_clip:
                clip_frames = []
            
            for frame in video_frames:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Only process face if we need face-related metrics and reference face was detected
                if (self.calculate_arc or self.calculate_cur or self.calculate_fid) and align_face_image is not None:
                    try:
                        align_face_frame, arcface_frame_embedding = self._process_image(frame_rgb)
                        
                        # Calculate requested scores
                        if self.calculate_cur:
                            cur_embedding_frame = self._inference(align_face_frame)
                            cur_score = max(0.0, self._batch_cosine_similarity(
                                cur_image_embedding, cur_embedding_frame).item())
                            cur_scores.append(cur_score)
                        
                        if self.calculate_arc:
                            arc_score = max(0.0, self._batch_cosine_similarity(
                                arcface_image_embedding, arcface_frame_embedding).item())
                            arc_scores.append(arc_score)
                        
                        if self.calculate_fid:
                            align_face_frame_pil = Image.fromarray(align_face_frame)
                            fake_image = self.transform(align_face_frame_pil).unsqueeze(0).to(self.device)
                            fake_activations = self._get_activations(fake_image)
                            fid_score = self._calculate_fid(real_activations, fake_activations)
                            fid_scores.append(fid_score)
                            
                    except Exception as e:
                        # print(f"Failed to process video frame face: {str(e)}")
                        fail_image_path = f"face_{os.path.basename(image)}_prompt_{current_clip_prompt[:10]}.png"
                        plt.imsave(f'/m2v_intern/liuhenglin/code/video_gen/flow_grpo/failure/{fail_image_path}', frame_rgb) 

                        # Append 0 for failed face detection in frame
                        if self.calculate_cur:
                            cur_scores.append(0.0)
                        if self.calculate_arc:
                            arc_scores.append(0.0)
                        if self.calculate_fid:
                            fid_scores.append(0.0)
                
                if self.calculate_clip:
                    clip_frames.append(frame_rgb)
            
            # Calculate average scores for requested metrics
            if self.calculate_cur and 'cur_score' not in results:
                results['cur_score'] = np.mean(cur_scores) if cur_scores else 0.0
            
            if self.calculate_arc and 'arc_score' not in results:
                results['arc_score'] = np.mean(arc_scores) if arc_scores else 0.0
            
            if self.calculate_fid and 'fid_score' not in results:
                results['fid_score'] = np.mean(fid_scores) if fid_scores else 0.0
            

            if self.calculate_clip and current_clip_prompt is not None and len(clip_frames) > 0:
                results['clip_score'] = self._compute_clip_score(clip_frames, current_clip_prompt)
            elif self.calculate_clip:
                results['clip_score'] = 0.0
            
            if reduction_mean:
                all_results.append(sum(results.values()) / len(results))
            else:
                all_results.append(results)
        
        return all_results

def sample_video_frames(video_path: str, num_frames: int = 16) -> List[np.ndarray]:
    """Sample frames from video"""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()
    return frames

        
if __name__ == "__main__":
    # Example usage with list inputs
    scorer = FaceSimilarityFIDScorer(
        device="cuda",
        calculate_cur=True,
        calculate_arc=True, # True, False
        calculate_fid=False,
        calculate_clip=False
    )
    
    # Example lists of inputs
    example_video_paths = [
        sample_video_frames("/m2v_intern/liuhenglin/code/video_gen/ConsisID/eval/output/1_stars_woman_Taylor_Swift_1.png--0/42_0000.mp4")
    ]
    example_image_paths = [
        load_image("/m2v_intern/liuhenglin/code/video_gen/data/eval/face_images/1_stars_woman_Taylor_Swift_1.png")
    ]
    clip_prompts = [
        "A {class_token} with a genuine smile tilts her head slightly toward the camera, her eyes reflecting the soft glow of the golden hour as the urban skyline forms a majestic backdrop; in a moment of spontaneity, a gentle breeze tousles her hair, adding a sense of vibrant energy to the serene atmosphere."
    ]
    
    # Calculate scores for all pairs
    scores = scorer(example_video_paths, example_image_paths, clip_prompt=clip_prompts)
    print(scores)
    # Print results for each pair
    # for i, score in enumerate(scores):
    #     print(f"\nResults for pair {i+1}:")
    #     if 'cur_score' in score:
    #         print(f"CurricularFace Similarity: {score['cur_score']:.4f}")
    #     if 'arc_score' in score:
    #         print(f"ArcFace Similarity: {score['arc_score']:.4f}")
    #     if 'fid_score' in score:
    #         print(f"FID Score: {score['fid_score']:.4f}")
    #     if 'clip_score' in score:
    #         print(f"CLIP Score: {score['clip_score']:.4f}")
    '''
    [{'cur_score': 0.7354490160942078, 'arc_score': 0.7119300328195095, 'fid_score': 48.8259539604187, 'clip_score': 25.454395294189453}]
    '''