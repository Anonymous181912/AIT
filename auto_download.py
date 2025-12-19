"""
Automatic image downloader for training dataset
Downloads real images from Pexels and AI-generated images from public sources
"""
import requests
import time
from pathlib import Path
from PIL import Image
import io
from typing import List
import random


class ImageDownloader:
    """Downloads images automatically for training"""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def download_real_images(self, count: int = 20) -> int:
        """
        Download real human portrait images from free public sources
        """
        print(f"\n📥 Downloading {count} real images from public sources...")
        
        downloaded = 0
        max_attempts = count * 3   # Try up to 3x the needed count
        attempts = 0
        
        # Get image URLs from multiple sources
        image_urls = self._get_real_image_urls(count * 2)  # Get extra URLs in case some fail
        
        for url in image_urls:
            if downloaded >= count:
                break
            
            if attempts >= max_attempts:
                break
            
            attempts += 1
            
            try:
                filename = self.output_dir / f"real_{downloaded + 1:03d}.jpg"
                
                # Skip if file already exists
                if filename.exists():
                    downloaded += 1
                    continue
                
                if self._download_image(url, filename):
                    downloaded += 1
                    print(f"  ✓ Downloaded {downloaded}/{count}: {filename.name}")
                    time.sleep(0.3)  # Be respectful to servers
                else:
                    # Silent fail, try next URL
                    pass
                    
            except Exception as e:
                # Silent fail, try next URL
                continue
        
        print(f"✓ Downloaded {downloaded} real images")
        return downloaded
    
    def download_ai_images(self, count: int = 20) -> int:
        """
        Download AI-generated face images from public sources
        """
        print(f"\n📥 Downloading {count} AI-generated images...")
        
        downloaded = 0
        
        # Get AI-generated image URLs
        ai_urls = self._get_ai_image_urls(count)
        
        for i, url in enumerate(ai_urls):
            if downloaded >= count:
                break
            
            try:
                filename = self.output_dir / f"fake_{downloaded + 1:03d}.jpg"
                if self._download_image(url, filename):
                    downloaded += 1
                    print(f"  ✓ Downloaded {downloaded}/{count}: {filename.name}")
                    time.sleep(0.5)  # Be respectful to servers
            except Exception as e:
                print(f"  ⚠ Failed to download AI image: {e}")
                continue
        
        print(f"✓ Downloaded {downloaded} AI-generated images")
        return downloaded
    
    def _download_image(self, url: str, filename: Path) -> bool:
        """Download a single image from URL"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            # Allow redirects (Unsplash Source redirects to actual image)
            response = requests.get(url, headers=headers, timeout=15, stream=True, allow_redirects=True)
            response.raise_for_status()
            
            # Check content type
            content_type = response.headers.get('content-type', '')
            if not content_type.startswith('image/'):
                # Try to load anyway, might be image without proper headers
                pass
            
            # Verify it's an image
            img_data = response.content
            if len(img_data) < 1000:  # Too small, probably not a real image
                return False
                
            img = Image.open(io.BytesIO(img_data))
            img = img.convert('RGB')
            
            # Skip if image is too small
            if img.width < 100 or img.height < 100:
                return False
            
            # Resize if too large (save space)
            max_size = 1024
            if img.width > max_size or img.height > max_size:
                img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
            # Save
            img.save(filename, 'JPEG', quality=85)
            return True
            
        except Exception as e:
            return False
    
    def _get_real_image_urls(self, count: int) -> List[str]:
        """
        Get URLs for real human portrait images
        Uses multiple free sources (Unsplash Source, Picsum, etc.)
        """
        urls = []
        
        # Method 1: Unsplash Source - free, no authentication needed
        # Note: Unsplash Source may redirect, so we handle that
        base_urls = [
            "https://source.unsplash.com/800x800/?portrait",
            "https://source.unsplash.com/800x800/?face",
            "https://source.unsplash.com/800x800/?person",
            "https://source.unsplash.com/800x800/?people",
            "https://source.unsplash.com/800x800/?human",
        ]
        
        # Method 2: Picsum Photos - free random images
        picsum_urls = [
            f"https://picsum.photos/800/800?random={random.randint(1, 1000)}"
            for _ in range(count // 2)
        ]
        
        # Method 3: Generate unique Unsplash URLs
        unsplash_urls = []
        for i in range(count):
            base = random.choice(base_urls)
            url = f"{base}&sig={random.randint(1000, 99999)}_{i}"
            unsplash_urls.append(url)
        
        # Combine sources
        urls = unsplash_urls + picsum_urls
        random.shuffle(urls)
        
        return urls[:count]
    
    def _get_ai_image_urls(self, count: int) -> List[str]:
        """
        Get URLs for AI-generated face images
        Uses public AI image generation services
        """
        urls = []
        
        # Method 1: This Person Does Not Exist API (free, generates AI faces)
        # Note: This service generates one image per request
        for i in range(count):
            # This Person Does Not Exist - generates unique AI faces
            url = f"https://thispersondoesnotexist.com/image?{random.randint(100000, 999999)}"
            urls.append(url)
        
        # Method 2: Alternative - use public AI-generated image dataset URLs
        # These are example URLs - in production, you might want to use a curated list
        alternative_urls = [
            # Add more public AI-generated image sources here if needed
        ]
        
        return urls


def ensure_dataset_exists(dataset_dir: str = "dataset", min_images: int = 20):
    """
    Ensure dataset has minimum required images, download if needed
    Returns True if dataset is ready, False otherwise
    """
    dataset_path = Path(dataset_dir)
    real_dir = dataset_path / "real"
    fake_dir = dataset_path / "fake"
    
    # Create directories
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)
    
    # Count existing images
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    real_count = sum(len(list(real_dir.glob(ext))) for ext in image_extensions)
    fake_count = sum(len(list(fake_dir.glob(ext))) for ext in image_extensions)
    
    print(f"📊 Current dataset: {real_count} real images, {fake_count} fake images")
    
    # Download real images if needed
    if real_count < min_images:
        needed = min_images - real_count
        print(f"\n⚠ Need {needed} more real images (have {real_count}, need {min_images})")
        print("🔄 Starting automatic download...")
        downloader = ImageDownloader(str(real_dir))
        downloaded = downloader.download_real_images(needed)
        real_count += downloaded
        print(f"✓ Now have {real_count} real images")
    else:
        print(f"✓ Real images: {real_count} (sufficient)")
    
    # Download AI images if needed
    if fake_count < min_images:
        needed = min_images - fake_count
        print(f"\n⚠ Need {needed} more AI-generated images (have {fake_count}, need {min_images})")
        print("🔄 Starting automatic download...")
        downloader = ImageDownloader(str(fake_dir))
        downloaded = downloader.download_ai_images(needed)
        fake_count += downloaded
        print(f"✓ Now have {fake_count} AI-generated images")
    else:
        print(f"✓ AI-generated images: {fake_count} (sufficient)")
    
    # Final check
    if real_count >= min_images and fake_count >= min_images:
        print(f"\n✅ Dataset ready: {real_count} real images, {fake_count} fake images")
        return True
    elif real_count > 0 and fake_count > 0:
        print(f"\n⚠ Dataset incomplete but usable: {real_count} real images, {fake_count} fake images")
        print("  Training will proceed with available images.")
        return True
    else:
        print(f"\n❌ Dataset insufficient: {real_count} real images, {fake_count} fake images")
        print("  Cannot proceed with training.")
        return False

