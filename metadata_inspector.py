"""
Metadata inspection module
Analyzes EXIF data and image metadata for authenticity indicators
Real camera images contain rich metadata, AI-generated images often lack it
"""
import os
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import exifread
import piexif
from typing import Dict, Optional, List
import numpy as np


class MetadataInspector:
    """Inspects image metadata for authenticity indicators"""
    
    def __init__(self):
        self.camera_tags = [
            'Make', 'Model', 'Software', 'DateTime', 'DateTimeOriginal',
            'ExifImageWidth', 'ExifImageHeight', 'Orientation',
            'XResolution', 'YResolution', 'ResolutionUnit'
        ]
        
        self.capture_tags = [
            'ExposureTime', 'FNumber', 'ISOSpeedRatings', 'FocalLength',
            'Flash', 'WhiteBalance', 'ColorSpace', 'ExposureMode',
            'MeteringMode', 'SceneCaptureType'
        ]
    
    def extract_exif_data(self, image_path: str) -> Dict[str, any]:
        """Extract EXIF data using multiple methods for robustness"""
        exif_data = {}
        
        # Method 1: Using PIL
        try:
            with Image.open(image_path) as img:
                exif_dict = img.getexif()
                if exif_dict:
                    for tag_id, value in exif_dict.items():
                        tag = TAGS.get(tag_id, tag_id)
                        exif_data[tag] = value
        except Exception as e:
            pass
        
        # Method 2: Using exifread (more detailed)
        try:
            with open(image_path, 'rb') as f:
                tags = exifread.process_file(f, details=True)
                for tag in tags.keys():
                    if tag not in ['JPEGThumbnail', 'TIFFThumbnail', 'Filename', 'EXIF MakerNote']:
                        exif_data[tag] = str(tags[tag])
        except Exception as e:
            pass
        
        # Method 3: Using piexif
        try:
            exif_dict = piexif.load(image_path)
            for ifd in ("0th", "Exif", "GPS", "1st"):
                if ifd in exif_dict:
                    for tag, value in exif_dict[ifd].items():
                        tag_name = piexif.TAGS[ifd][tag]["name"]
                        exif_data[tag_name] = value
        except Exception as e:
            pass
        
        return exif_data
    
    def analyze_metadata_completeness(self, exif_data: Dict[str, any]) -> Dict[str, float]:
        """
        Analyze how complete the metadata is
        Real camera images have rich metadata, AI-generated often lack it
        """
        completeness = {
            'has_camera_info': 0.0,
            'has_capture_settings': 0.0,
            'has_datetime': 0.0,
            'has_software': 0.0,
            'total_metadata_count': len(exif_data),
        }
        
        # Check for camera information
        has_make = any('make' in str(k).lower() for k in exif_data.keys())
        has_model = any('model' in str(k).lower() for k in exif_data.keys())
        completeness['has_camera_info'] = 1.0 if (has_make or has_model) else 0.0
        
        # Check for capture settings
        has_exposure = any('exposure' in str(k).lower() for k in exif_data.keys())
        has_iso = any('iso' in str(k).lower() for k in exif_data.keys())
        has_focal = any('focal' in str(k).lower() for k in exif_data.keys())
        completeness['has_capture_settings'] = 1.0 if (has_exposure or has_iso or has_focal) else 0.0
        
        # Check for datetime
        has_datetime = any('datetime' in str(k).lower() for k in exif_data.keys())
        completeness['has_datetime'] = 1.0 if has_datetime else 0.0
        
        # Check for software
        has_software = any('software' in str(k).lower() for k in exif_data.keys())
        completeness['has_software'] = 1.0 if has_software else 0.0
        
        return completeness
    
    def detect_metadata_anomalies(self, exif_data: Dict[str, any]) -> Dict[str, float]:
        """
        Detect anomalies in metadata that suggest AI generation
        """
        anomalies = {
            'suspicious_software': 0.0,
            'missing_camera_info': 0.0,
            'inconsistent_resolution': 0.0,
            'unusual_datetime': 0.0,
        }
        
        # Check for AI generation software
        software_tags = [str(v).lower() for k, v in exif_data.items() 
                        if 'software' in str(k).lower() or 'program' in str(k).lower()]
        
        ai_keywords = ['stable diffusion', 'midjourney', 'dall-e', 'dalle', 
                      'generated', 'ai', 'synthetic', 'gan']
        for software in software_tags:
            if any(keyword in software for keyword in ai_keywords):
                anomalies['suspicious_software'] = 1.0
                break
        
        # Check if camera info is missing (common in AI-generated)
        has_make = any('make' in str(k).lower() for k in exif_data.keys())
        has_model = any('model' in str(k).lower() for k in exif_data.keys())
        if not (has_make or has_model):
            anomalies['missing_camera_info'] = 1.0
        
        # Check resolution consistency
        width_tags = [v for k, v in exif_data.items() 
                     if 'width' in str(k).lower() or 'imagewidth' in str(k).lower()]
        height_tags = [v for k, v in exif_data.items() 
                      if 'height' in str(k).lower() or 'imagelength' in str(k).lower()]
        
        if width_tags and height_tags:
            # Check if resolution values are consistent
            widths = [int(v) for v in width_tags if str(v).isdigit()]
            heights = [int(v) for v in height_tags if str(v).isdigit()]
            if widths and heights:
                if len(set(widths)) > 1 or len(set(heights)) > 1:
                    anomalies['inconsistent_resolution'] = 1.0
        
        return anomalies
    
    def extract_metadata_signature(self, image_path: str) -> np.ndarray:
        """
        Extract compact metadata signature for classification
        """
        exif_data = self.extract_exif_data(image_path)
        completeness = self.analyze_metadata_completeness(exif_data)
        anomalies = self.detect_metadata_anomalies(exif_data)
        
        signature = []
        
        # Completeness features
        signature.append(completeness['has_camera_info'])
        signature.append(completeness['has_capture_settings'])
        signature.append(completeness['has_datetime'])
        signature.append(completeness['has_software'])
        signature.append(min(completeness['total_metadata_count'] / 50.0, 1.0))  # Normalized
        
        # Anomaly features
        signature.append(anomalies['suspicious_software'])
        signature.append(anomalies['missing_camera_info'])
        signature.append(anomalies['inconsistent_resolution'])
        signature.append(anomalies['unusual_datetime'])
        
        # Additional metadata statistics
        # Check for GPS data (real photos often have location)
        has_gps = any('gps' in str(k).lower() for k in exif_data.keys())
        signature.append(1.0 if has_gps else 0.0)
        
        # Check for thumbnail (real cameras often embed thumbnails)
        has_thumbnail = 'JPEGThumbnail' in exif_data or 'TIFFThumbnail' in exif_data
        signature.append(1.0 if has_thumbnail else 0.0)
        
        return np.array(signature, dtype=np.float32)

