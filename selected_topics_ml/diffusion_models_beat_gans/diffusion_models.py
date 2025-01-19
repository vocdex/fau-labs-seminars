import torch
import torchvision.transforms as transforms
import os
import math
from PIL import Image

def resize_image(image_path, scale_factor):
    """
    Resize image while keeping aspect ratio
    
    Args:
        image_path (str): Path to input image
        scale_factor (float): Scaling factor
        
    Returns:
        str: Path to resized image
    """
    img = Image.open(image_path)
    width, height = img.size
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    # Replace ANTIALIAS with Lanczos resampling
    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    new_path = image_path.replace('.png', f'_resized_{scale_factor}.png')
    img.save(new_path)
    return new_path

def make_gif(frames_dir, delete_frames=True):
    from moviepy.editor import ImageSequenceClip
    from natsort import natsorted
    import glob
    frame_files = natsorted(glob.glob(os.path.join(frames_dir, "*.png")))
    # Reduce image size while keeping aspect ratio
    frame_files = [resize_image(f, 0.5) for f in frame_files]
    
    scheduler = os.path.basename(frames_dir)

    clip = ImageSequenceClip(frame_files, fps=10)
    clip.write_gif(f'{frames_dir}/{scheduler}_noising.gif')
    if delete_frames:
        for f in frame_files:
            os.remove(f)

class ForwardDiffusionProcess:
    def __init__(self, image_path, num_steps=1000, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize forward diffusion process
        
        Args:
            image_path (str): Path to the input image
            num_steps (int): Number of diffusion steps
            device (str): Device to run the computation on
        """
        self.device = device
        self.num_steps = num_steps
        
        self.original_image = self.load_and_preprocess_image(image_path)
        
        self.base_output_dir = 'diffusion_steps'
        os.makedirs(self.base_output_dir, exist_ok=True)
    
    def load_and_preprocess_image(self, image_path):
        """
        Load image and preprocess for diffusion
        
        Args:
            image_path (str): Path to input image
        
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        img = Image.open(image_path).convert('RGB')
        
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        
        image = transform(img).unsqueeze(0).to(self.device)
        return image
    
    def cosine_beta_schedule(self, s=0.008):
        """
        Cosine noise scheduler
        
        Args:
            s (float): Offset to prevent division by zero
        
        Returns:
            torch.Tensor: Beta values for noise schedule
        """
        steps = self.num_steps + 1
        x = torch.linspace(0, self.num_steps, steps, dtype=torch.float64)/self.num_steps
        alphas_cumprod = torch.cos((x+s)/(1+s)*math.pi * 0.5)**2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)
    
    def linear_beta_schedule(self):
        """
        Linear noise scheduler
        
        Returns:
            torch.Tensor: Beta values for noise schedule
        """
        scale = 1000/ self.num_steps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        betas = torch.linspace(beta_start, beta_end, self.num_steps, dtype=torch.float64)
        return betas
    
    def forward_diffusion(self, scheduler_type='cosine'):
        """
        Perform forward diffusion process
        
        Args:
            scheduler_type (str): Type of noise scheduler ('cosine' or 'linear')
        """
        from PIL import Image, ImageDraw, ImageFont
        import numpy as np
        
        self.output_dir = os.path.join(self.base_output_dir, f'{scheduler_type}')
        os.makedirs(self.output_dir, exist_ok=True)
        if scheduler_type == 'cosine':
            betas = self.cosine_beta_schedule().to(self.device)
        else:
            betas = self.linear_beta_schedule().to(self.device)
        
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        noisy_image = self.original_image.clone()
        for t in range(self.num_steps):
            if t % 10==0:
                # Add noise
                noise = torch.randn_like(noisy_image)
                sqrt_alpha_cumprod = torch.sqrt(alphas_cumprod[t])
                sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - alphas_cumprod[t])
                
                noisy_image = sqrt_alpha_cumprod * noisy_image + sqrt_one_minus_alpha_cumprod * noise
                
                img_array = noisy_image[0].cpu().permute(1, 2, 0).numpy()
                img_array = (img_array * 255).clip(0, 255).astype(np.uint8)
                pil_image = Image.fromarray(img_array)
                
                draw = ImageDraw.Draw(pil_image)
                text = f"Step: {t}/{self.num_steps}"
                try:
                    font = ImageFont.truetype("Arial.ttf", 50)
                except:
                    font = ImageFont.load_default()
                    
                x, y = 20, 15  # Position of text
                for offset in [(1,1), (-1,-1), (1,-1), (-1,1)]:
                    draw.text((x+offset[0], y+offset[1]), text, font=font, fill='black')
                draw.text((x, y), text, font=font, fill='white')
                pil_image.save(f'{self.output_dir}/step_{t}.png')


if __name__ == '__main__':
    image_path = 'topological_lady.jpeg'
    
    diffusion = ForwardDiffusionProcess(image_path, num_steps=1000)
    
    diffusion.forward_diffusion(scheduler_type='cosine')
    
    diffusion.forward_diffusion(scheduler_type='linear')

    make_gif('diffusion_steps/cosine')
    make_gif('diffusion_steps/linear')