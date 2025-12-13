"""
Inference script for deepfake detection
"""
import argparse
import torch
from pathlib import Path
import json
from pipeline import DeepfakeDetectionPipeline
from config import MODEL_DIR


def main():
    parser = argparse.ArgumentParser(description="Deepfake Detection Inference")
    parser.add_argument("--image", type=str, required=True, help="Path to image file")
    parser.add_argument("--model", type=str, default=None, 
                       help="Path to model file (default: best_model.pth)")
    parser.add_argument("--device", type=str, default="cpu", 
                       help="Device to use (cpu/cuda)")
    parser.add_argument("--explain", action="store_true", 
                       help="Show detailed explanation of prediction")
    parser.add_argument("--batch", type=str, default=None,
                       help="Path to directory or JSON file with multiple images")
    
    args = parser.parse_args()
    
    # Determine model path
    if args.model is None:
        model_path = Path(MODEL_DIR) / "best_model.pth"
    else:
        model_path = args.model
    
    if not Path(model_path).exists():
        print(f"Error: Model file not found at {model_path}")
        print("Please train the model first using train.py")
        return
    
    # Initialize pipeline
    print(f"Loading model from {model_path}...")
    pipeline = DeepfakeDetectionPipeline(
        model_path=str(model_path),
        device=args.device
    )
    
    # Single image inference
    if args.batch is None:
        if not Path(args.image).exists():
            print(f"Error: Image file not found at {args.image}")
            return
        
        print(f"\nAnalyzing image: {args.image}")
        
        if args.explain:
            result = pipeline.explain_prediction(args.image)
            print("\n" + "="*50)
            print("PREDICTION EXPLANATION")
            print("="*50)
            print(f"Prediction: {result['prediction']}")
            print(f"Confidence: {result['confidence']:.4f}")
            print("\nFeature Contributions:")
            for feature, importance in result['feature_contributions'].items():
                print(f"  {feature}: {importance:.4f}")
            print("\nFeature Statistics:")
            for feature, stat in result['feature_statistics'].items():
                print(f"  {feature}: {stat:.4f}")
        else:
            result = pipeline.predict(args.image)
            print("\n" + "="*50)
            print("PREDICTION RESULT")
            print("="*50)
            print(f"Image: {args.image}")
            print(f"Prediction: {result['prediction']}")
            print(f"Confidence: {result['confidence']:.4f}")
            print(f"Real Probability: {result['real_probability']:.4f}")
            print(f"AI-Generated Probability: {result['ai_generated_probability']:.4f}")
    
    # Batch inference
    else:
        image_paths = []
        
        # Check if it's a directory
        if Path(args.batch).is_dir():
            image_paths = list(Path(args.batch).glob("*.jpg")) + \
                         list(Path(args.batch).glob("*.png")) + \
                         list(Path(args.batch).glob("*.jpeg"))
            image_paths = [str(p) for p in image_paths]
        # Check if it's a JSON file
        elif Path(args.batch).exists() and args.batch.endswith('.json'):
            with open(args.batch, 'r') as f:
                data = json.load(f)
                image_paths = data.get('images', [])
        else:
            print(f"Error: Invalid batch input: {args.batch}")
            return
        
        print(f"\nProcessing {len(image_paths)} images...")
        results = pipeline.predict_batch(image_paths)
        
        # Print results
        print("\n" + "="*50)
        print("BATCH PREDICTION RESULTS")
        print("="*50)
        for result in results:
            print(f"\n{result['image_path']}")
            print(f"  Prediction: {result['prediction']}")
            print(f"  Confidence: {result['confidence']:.4f}")
        
        # Save results
        output_path = "batch_results.json"
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()

