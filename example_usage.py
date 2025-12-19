"""
Example usage script demonstrating the deepfake detection system
"""
import os
from pathlib import Path
from pipeline import DeepfakeDetectionPipeline
from background_features import BackgroundFeatureExtractor
from face_integration import HybridDeepfakeDetector


def example_basic_detection():
    """Basic single image detection"""
    print("="*60)
    print("Example 1: Basic Detection")
    print("="*60)
    
    # Initialize pipeline (without model for feature extraction demo)
    pipeline = DeepfakeDetectionPipeline(device="cpu")
    
    # Example: Extract features from an image
    # Replace with your image path
    image_path = "example_image.jpg"
    
    if not Path(image_path).exists():
        print(f"Image not found: {image_path}")
        print("Please provide a valid image path")
        return
    
    # Extract features
    print(f"\nExtracting features from: {image_path}")
    feature_extractor = BackgroundFeatureExtractor()
    features = feature_extractor.extract_all_features(image_path)
    
    print(f"\nFeature Dimensions:")
    print(f"  Frequency features: {len(features['frequency'])}")
    print(f"  Noise features: {len(features['noise'])}")
    print(f"  Metadata features: {len(features['metadata'])}")
    print(f"  Total: {len(features['frequency']) + len(features['noise']) + len(features['metadata'])}")
    
    print(f"\nFeature Statistics:")
    print(f"  Frequency mean: {features['frequency'].mean():.4f}")
    print(f"  Noise mean: {features['noise'].mean():.4f}")
    print(f"  Metadata mean: {features['metadata'].mean():.4f}")


def example_prediction():
    """Example prediction with trained model"""
    print("\n" + "="*60)
    print("Example 2: Prediction with Trained Model")
    print("="*60)
    
    model_path = "models/best_model.pth"
    image_path = "example_image.jpg"
    
    if not Path(model_path).exists():
        print(f"Model not found: {model_path}")
        print("Please train the model first using: python train.py")
        return
    
    if not Path(image_path).exists():
        print(f"Image not found: {image_path}")
        return
    
    # Initialize pipeline with trained model
    pipeline = DeepfakeDetectionPipeline(model_path=model_path, device="cpu")
    
    # Make prediction
    result = pipeline.predict(image_path)
    
    print(f"\nPrediction Results:")
    print(f"  Image: {image_path}")
    print(f"  Prediction: {result['prediction']}")
    print(f"  Confidence: {result['confidence']:.4f}")
    print(f"  Real Probability: {result['real_probability']:.4f}")
    print(f"  AI-Generated Probability: {result['ai_generated_probability']:.4f}")


def example_explanation():
    """Example with detailed explanation"""
    print("\n" + "="*60)
    print("Example 3: Detailed Explanation")
    print("="*60)
    
    model_path = "models/best_model.pth"
    image_path = "example_image.jpg"
    
    if not Path(model_path).exists() or not Path(image_path).exists():
        print("Model or image not found. Skipping...")
        return
    
    pipeline = DeepfakeDetectionPipeline(model_path=model_path, device="cpu")
    explanation = pipeline.explain_prediction(image_path)
    
    print(f"\nExplanation:")
    print(f"  Prediction: {explanation['prediction']}")
    print(f"  Confidence: {explanation['confidence']:.4f}")
    print(f"\n  Feature Contributions:")
    for feature, importance in explanation['feature_contributions'].items():
        print(f"    {feature}: {importance:.4f}")
    print(f"\n  Feature Statistics:")
    for feature, stat in explanation['feature_statistics'].items():
        print(f"    {feature}: {stat:.4f}")


def example_hybrid_detection():
    """Example using hybrid background + face detection"""
    print("\n" + "="*60)
    print("Example 4: Hybrid Detection (Background + Face)")
    print("="*60)
    
    model_path = "models/best_model.pth"
    image_path = "example_image.jpg"
    
    if not Path(model_path).exists() or not Path(image_path).exists():
        print("Model or image not found. Skipping...")
        return
    
    # Initialize hybrid detector
    bg_pipeline = DeepfakeDetectionPipeline(model_path=model_path, device="cpu")
    hybrid = HybridDeepfakeDetector(bg_pipeline)
    
    # Detect
    result = hybrid.detect(image_path)
    
    print(f"\nHybrid Detection Results:")
    print(f"  Final Prediction: {result['prediction']}")
    print(f"  Confidence: {result['confidence']:.4f}")
    print(f"  Combined Score: {result['combined_score']:.4f}")
    print(f"  Face Detected: {result['face_analysis']['face_detected']}")
    print(f"  Background Analysis: {result['background_analysis']['prediction']}")


def example_feature_analysis():
    """Example showing how different features distinguish real vs fake"""
    print("\n" + "="*60)
    print("Example 5: Feature Analysis")
    print("="*60)
    
    feature_extractor = BackgroundFeatureExtractor()
    
    # Analyze multiple images
    real_images = list(Path("data/test/real").glob("*.jpg"))[:3]
    fake_images = list(Path("data/test/fake").glob("*.jpg"))[:3]
    
    if not real_images or not fake_images:
        print("Test images not found. Please prepare test dataset.")
        return
    
    print("\nAnalyzing Real Images:")
    real_features = []
    for img_path in real_images:
        features = feature_extractor.extract_unified_signature(str(img_path))
        real_features.append(features)
        print(f"  {img_path.name}: freq={features[:20].mean():.4f}, noise={features[20:34].mean():.4f}")
    
    print("\nAnalyzing AI-Generated Images:")
    fake_features = []
    for img_path in fake_images:
        features = feature_extractor.extract_unified_signature(str(img_path))
        fake_features.append(features)
        print(f"  {img_path.name}: freq={features[:20].mean():.4f}, noise={features[20:34].mean():.4f}")
    
    if real_features and fake_features:
        import numpy as np
        real_mean = np.mean(real_features, axis=0)
        fake_mean = np.mean(fake_features, axis=0)
        
        print(f"\nFeature Differences (Real vs Fake):")
        print(f"  Frequency features diff: {np.abs(real_mean[:20] - fake_mean[:20]).mean():.4f}")
        print(f"  Noise features diff: {np.abs(real_mean[20:34] - fake_mean[20:34]).mean():.4f}")
        print(f"  Metadata features diff: {np.abs(real_mean[34:] - fake_mean[34:]).mean():.4f}")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Deepfake Detection System - Example Usage")
    print("="*60)
    
    # Run examples
    example_basic_detection()
    
    # Uncomment to run other examples (requires trained model and data)
    # example_prediction()
    # example_explanation()
    # example_hybrid_detection()
    # example_feature_analysis()
    
    print("\n" + "="*60)
    print("Examples completed!")
    print("="*60)
    print("\nNext steps:")
    print("1. Prepare your dataset in data/train/ and data/test/")
    print("2. Train the model: python train.py")
    print("3. Run inference: python infer.py --image your_image.jpg")
    print("4. Evaluate: python evaluate.py")

