#!/usr/bin/env python3
"""
5G NTN MCS Recommendation Demo Script

This script demonstrates how to use the trained XGBoost model for:
1. Simple inference: predict pass probability for given context + MCS
2. MCS optimization: find optimal MCS using different objectives
3. Realistic scenarios: showcase model performance across different conditions

"""

import json
import numpy as np
import xgboost as xgb
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from mcs_tables import spectral_efficiency


class MCSRecommender:
    """5G NTN MCS Recommendation System

    Args:
        model_path: Path to saved XGBoost model
        meta_path: Path to metadata JSON
        device: Optional prediction device hint ("cpu", "cuda", or None to leave as-is)
    """
    
    def __init__(self, model_path: str = "models/xgb_mcs_pass.json", 
                 meta_path: str = "models/model_meta.json",
                 device: Optional[str] = None):
        self.model_path = Path(model_path)
        self.meta_path = Path(meta_path)
        self._device = None
        self._load_model()
        self._load_metadata()
        if device is not None:
            self.set_device(device)
    
    def _load_model(self):
        """Load the trained XGBoost model"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}. Run training first.")
        
        self.model = xgb.Booster()
        self.model.load_model(str(self.model_path))
        print(f"✓ Loaded XGBoost model from {self.model_path}")

    def set_device(self, device: str):
        """Set prediction device if supported by installed XGBoost.

        device: "cpu" or "cuda"
        """
        dev = device.lower().strip()
        if dev not in ("cpu", "cuda"):
            raise ValueError("device must be 'cpu' or 'cuda'")
        try:
            # XGBoost >= 2.0 supports 'device'
            self.model.set_param({"device": dev})
            self._device = dev
            print(f"Inference device set to {dev}")
        except Exception:
            # Fallback for older versions
            predictor = "gpu_predictor" if dev == "cuda" else "cpu_predictor"
            try:
                self.model.set_param({"predictor": predictor})
                self._device = dev
                print(f"Inference predictor set to {predictor}")
            except Exception as e:
                print(f"Could not set device/predictor: {e}")
                self._device = None
    
    def _load_metadata(self):
        """Load model metadata including features and calibrated threshold"""
        if not self.meta_path.exists():
            raise FileNotFoundError(f"Metadata not found at {self.meta_path}")
        
        with open(self.meta_path, 'r') as f:
            self.meta = json.load(f)
        
        self.features = self.meta["features"]
        self.threshold = self.meta.get("threshold", 0.5)
        self.calibrate_target = self.meta.get("calibrate_target", 0.1)
        
        print(f"✓ Loaded metadata: {len(self.features)} features, threshold={self.threshold:.3f}")
        print(f"  Features: {', '.join(self.features)}")
    
    def predict_pass_probability(self, context: Dict[str, float], mcs: int) -> float:
        """
        Predict pass probability for a given context and MCS
        
        Args:
            context: Dictionary with network context parameters
            mcs: MCS index to evaluate
        
        Returns:
            Pass probability [0, 1]
        """
        # Build feature vector
        feature_values = []
        for feat in self.features:
            if feat == "mcs":
                feature_values.append(float(mcs))
            elif feat in context:
                feature_values.append(float(context[feat]))
            else:
                raise ValueError(f"Missing required feature: {feat}")
        
        # Create DMatrix and predict
        X = np.array(feature_values).reshape(1, -1).astype(np.float32)
        dmat = xgb.DMatrix(X)
        prob = self.model.predict(dmat)[0]
        
        return float(prob)
    
    def recommend_mcs_threshold(self, context: Dict[str, float], 
                               mcs_range: Tuple[int, int] = (0, 27)) -> Tuple[int, float]:
        """
        Recommend highest MCS meeting the reliability threshold
        
        Args:
            context: Network context parameters
            mcs_range: (min_mcs, max_mcs) to consider
        
        Returns:
            (recommended_mcs, pass_probability)
        """
        best_mcs = mcs_range[0]
        best_prob = 0.0
        
        for mcs in range(mcs_range[0], mcs_range[1] + 1):
            prob = self.predict_pass_probability(context, mcs)
            if prob >= self.threshold and mcs >= best_mcs:
                best_mcs = mcs
                best_prob = prob
        
        return best_mcs, best_prob
    
    def recommend_mcs_throughput(self, context: Dict[str, float], 
                                mcs_range: Tuple[int, int] = (0, 27)) -> Tuple[int, float, float]:
        """
        Recommend MCS maximizing expected throughput (spectral_efficiency x pass_prob)
        
        Args:
            context: Network context parameters
            mcs_range: (min_mcs, max_mcs) to consider
        
        Returns:
            (recommended_mcs, pass_probability, expected_throughput_score)
        """
        best_mcs = mcs_range[0]
        best_prob = 0.0
        best_score = 0.0
        
        for mcs in range(mcs_range[0], mcs_range[1] + 1):
            prob = self.predict_pass_probability(context, mcs)
            eff = spectral_efficiency(mcs)
            score = prob * eff
            
            if score > best_score:
                best_mcs = mcs
                best_prob = prob
                best_score = score
        
        return best_mcs, best_prob, best_score

    def recommend_mcs_tput_constrained(self, context: Dict[str, float], 
                                       mcs_range: Tuple[int, int] = (0, 27),
                                       tau: Optional[float] = None) -> Tuple[int, float, float]:
        """
        Throughput-constrained objective: maximize spectral_efficiency x P(pass)
        subject to P(pass) ≥ τ. If no MCS meets τ, fallback to the argmax of P(pass).

        Returns: (mcs, prob, score)
        """
        thr = self.threshold if tau is None else float(tau)
        best_mcs = mcs_range[0]
        best_prob = 0.0
        best_score = -1.0
        fallback_best_p = 0.0
        fallback_best_mcs = mcs_range[0]
        fallback_best_score = 0.0
        for mcs in range(mcs_range[0], mcs_range[1] + 1):
            prob = self.predict_pass_probability(context, mcs)
            eff = spectral_efficiency(mcs)
            score = prob * eff
            if prob >= thr and score > best_score:
                best_mcs = mcs
                best_prob = prob
                best_score = score
            if prob > fallback_best_p:
                fallback_best_p = prob
                fallback_best_mcs = mcs
                fallback_best_score = score
        if best_score >= 0:
            return best_mcs, best_prob, best_score
        # Fallback when no candidate meets τ
        return fallback_best_mcs, fallback_best_p, fallback_best_score


class ThroughputPolicy:
    """Simple helper to run the throughput-based system from Python.

    Usage:
        policy = ThroughputPolicy(device="cpu")
        rec = policy.recommend(context, guardrail=0.6)  # optional reliability guardrail
        # rec -> {"mcs": int, "prob": float, "eff": float, "expected": float}

        batch = policy.batch_recommend(list_of_contexts, guardrail=0.6)

    Notes:
        - expected throughput is computed as: P(pass | context, MCS) x spectral_efficiency(MCS)
        - if guardrail is provided, only MCS with P(pass) ≥ guardrail are considered.
    """

    def __init__(self, model_path: str = "models/xgb_mcs_pass.json",
                 meta_path: str = "models/model_meta.json",
                 device: Optional[str] = None):
        self.inner = MCSRecommender(model_path, meta_path, device=device)

    def expected_throughput(self, context: Dict[str, float], mcs: int) -> float:
        p = self.inner.predict_pass_probability(context, mcs)
        return p * spectral_efficiency(mcs)

    def _scores_for_context(self, context: Dict[str, float], mcs_values: List[int]):
        # Build a feature row per candidate MCS to amortize DMatrix creation
        import numpy as np
        feats = self.inner.features
        X = np.zeros((len(mcs_values), len(feats)), dtype=np.float32)
        for i, m in enumerate(mcs_values):
            for j, f in enumerate(feats):
                if f == "mcs":
                    X[i, j] = float(m)
                else:
                    X[i, j] = float(context[f])
        dmat = xgb.DMatrix(X)
        probs = self.inner.model.predict(dmat)
        effs = np.array([spectral_efficiency(m) for m in mcs_values], dtype=np.float32)
        scores = probs * effs
        return probs, effs, scores

    def recommend(self, context: Dict[str, float],
                  mcs_range: Tuple[int, int] = (0, 27),
                  guardrail: Optional[float] = None) -> Dict[str, float]:
        mcs_values = list(range(mcs_range[0], mcs_range[1] + 1))
        probs, effs, scores = self._scores_for_context(context, mcs_values)
        if guardrail is not None:
            mask = probs >= float(guardrail)
            if not mask.any():
                # Fall back to the most conservative MCS
                idx = 0
            else:
                # Among feasible, pick argmax of throughput
                idx = int(np.argmax(scores * mask))
        else:
            idx = int(np.argmax(scores))
        mcs = int(mcs_values[idx])
        return {
            "mcs": mcs,
            "prob": float(probs[idx]),
            "eff": float(effs[idx]),
            "expected": float(scores[idx]),
        }

    def batch_recommend(self, contexts: List[Dict[str, float]],
                        mcs_range: Tuple[int, int] = (0, 27),
                        guardrail: Optional[float] = None) -> List[Dict[str, float]]:
        results: List[Dict[str, float]] = []
        mcs_values = list(range(mcs_range[0], mcs_range[1] + 1))
        for ctx in contexts:
            results.append(self.recommend(ctx, mcs_range=mcs_range, guardrail=guardrail))
        return results


def create_sample_contexts() -> List[Tuple[str, Dict[str, float]]]:
    """Create realistic 5G NTN scenarios for demonstration"""
    return [
        ("High Elevation (Good Conditions)", {
            "slot_percent": 0.8,
            "slot": 5.0,
            "ele_angle": 75.0,  # High elevation - good link
            "pathloss": 145.0,  # Lower pathloss
            "snr": 15.0,        # Good SNR
            "cqi": 12.0,        # High CQI
            "window": 100.0,    # Short averaging window
            "target_bler": 0.01 # Strict BLER target
        }),
        
        ("Medium Elevation (Moderate Conditions)", {
            "slot_percent": 0.6,
            "slot": 3.0,
            "ele_angle": 45.0,  # Medium elevation
            "pathloss": 155.0,  # Moderate pathloss
            "snr": 8.0,         # Moderate SNR
            "cqi": 8.0,         # Medium CQI
            "window": 200.0,    # Medium averaging window
            "target_bler": 0.05 # Moderate BLER target
        }),
        
        ("Low Elevation (Challenging Conditions)", {
            "slot_percent": 0.3,
            "slot": 1.0,
            "ele_angle": 15.0,  # Low elevation - challenging
            "pathloss": 165.0,  # Higher pathloss
            "snr": 2.0,         # Poor SNR
            "cqi": 4.0,         # Low CQI
            "window": 500.0,    # Long averaging window
            "target_bler": 0.1  # Relaxed BLER target
        })
    ]


def demonstrate_simple_inference(recommender: MCSRecommender):
    """Demonstrate basic pass probability prediction"""
    print("\n" + "="*60)
    print("1. SIMPLE INFERENCE DEMONSTRATION")
    print("="*60)
    
    # Example context
    context = {
        "slot_percent": 0.7,
        "slot": 4.0,
        "ele_angle": 60.0,
        "pathloss": 150.0,
        "snr": 10.0,
        "cqi": 9.0,
        "window": 150.0,
        "target_bler": 0.01
    }
    
    print("Network Context:")
    for param, value in context.items():
        print(f"  {param}: {value}")
    
    print(f"\nPass Probability Predictions:")
    print(f"{'MCS':<3} {'Spectral Eff':<12} {'Pass Prob':<10} {'Status'}")
    print("-" * 40)
    
    for mcs in [5, 10, 15, 20, 25]:
        prob = recommender.predict_pass_probability(context, mcs)
        eff = spectral_efficiency(mcs)
        status = "✓ Pass" if prob >= recommender.threshold else "✗ Fail"
        print(f"{mcs:<3} {eff:<12.3f} {prob:<10.3f} {status}")


def demonstrate_optimization(recommender: MCSRecommender):
    """Demonstrate MCS optimization strategies"""
    print("\n" + "="*60)
    print("2. MCS OPTIMIZATION DEMONSTRATION")
    print("="*60)
    
    contexts = create_sample_contexts()
    
    for scenario_name, context in contexts:
        print(f"\nScenario: {scenario_name}")
        print("-" * 50)
        
        # Threshold-based optimization
        mcs_thresh, prob_thresh = recommender.recommend_mcs_threshold(context)
        
        # Throughput-based optimization  
        mcs_tput, prob_tput, score_tput = recommender.recommend_mcs_throughput(context)
        
        # Baseline CQI mapping
        cqi = context.get("cqi", 9.0)
        mcs_baseline = max(0, min(27, int(round(cqi))))
        prob_baseline = recommender.predict_pass_probability(context, mcs_baseline)
        
        print(f"Key Parameters: ElevAngle={context['ele_angle']:.1f}°, SNR={context['snr']:.1f}dB, CQI={context['cqi']:.0f}")
        print(f"Model Threshold: {recommender.threshold:.3f} (target violation: {recommender.calibrate_target:.1%})")
        print()
        print(f"{'Strategy':<20} {'MCS':<4} {'Pass Prob':<10} {'Spectral Eff':<12} {'Expected Score'}")
        print("-" * 70)
        
        # Threshold strategy
        eff_thresh = spectral_efficiency(mcs_thresh)
        score_thresh = prob_thresh * eff_thresh
        print(f"{'Threshold-based':<20} {mcs_thresh:<4} {prob_thresh:<10.3f} {eff_thresh:<12.3f} {score_thresh:<12.3f}")
        
        # Throughput strategy
        eff_tput = spectral_efficiency(mcs_tput)
        print(f"{'Throughput-based':<20} {mcs_tput:<4} {prob_tput:<10.3f} {eff_tput:<12.3f} {score_tput:<12.3f}")
        
        # Baseline
        eff_baseline = spectral_efficiency(mcs_baseline)
        score_baseline = prob_baseline * eff_baseline
        print(f"{'CQI Baseline':<20} {mcs_baseline:<4} {prob_baseline:<10.3f} {eff_baseline:<12.3f} {score_baseline:<12.3f}")


def demonstrate_model_insights(recommender: MCSRecommender):
    """Display model performance and metadata insights"""
    print("\n" + "="*60)
    print("3. MODEL PERFORMANCE INSIGHTS")
    print("="*60)
    
    print(f"Model Metadata:")
    print(f"  Training samples: {recommender.meta.get('train_rows', 'N/A'):,}")
    print(f"  Test samples: {recommender.meta.get('test_rows', 'N/A'):,}")
    print(f"  Device used: {recommender.meta.get('device_used', 'N/A')}")
    print(f"  Best iteration: {recommender.meta.get('best_iteration', 'N/A')}")
    
    print(f"\nThreshold Calibration:")
    print(f"  Calibrated threshold: {recommender.threshold:.4f}")
    print(f"  Target violation rate: {recommender.calibrate_target:.1%}")
    print(f"  → This means ~{recommender.calibrate_target:.1%} of accepted transmissions may fail")
    
    # Load and display metrics if available
    metrics_path = Path("models/metrics.json")
    if metrics_path.exists():
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        print(f"\nModel Performance on Test Set:")
        print(f"  Log Loss: {metrics.get('logloss', 'N/A'):.4f}")
        print(f"  Accuracy: {metrics.get('accuracy', 'N/A'):.3f}")


def demonstrate_practical_usage():
    """Show practical usage patterns and integration examples"""
    print("\n" + "="*60)
    print("4. PRACTICAL USAGE PATTERNS")
    print("="*60)
    
    print("""
# Single Context Recommendation
recommender = MCSRecommender()
context = {
    "slot_percent": 0.8, "slot": 5.0, "ele_angle": 70.0,
    "pathloss": 148.0, "snr": 12.0, "cqi": 10.0, 
    "window": 100.0, "target_bler": 0.01
}

# Get threshold-based recommendation
mcs, prob = recommender.recommend_mcs_threshold(context)
print(f"Recommended MCS: {mcs}, Pass Probability: {prob:.3f}")

# Get throughput-optimized recommendation  
mcs, prob, score = recommender.recommend_mcs_throughput(context)
print(f"Throughput-optimal MCS: {mcs}, Expected Score: {score:.3f}")

# Direct probability prediction
prob = recommender.predict_pass_probability(context, mcs=15)
print(f"MCS 15 pass probability: {prob:.3f}")

# Batch Processing Example:
# contexts = load_contexts_from_csv("network_measurements.csv")
# recommendations = []
# for ctx in contexts:
#     mcs, prob = recommender.recommend_mcs_threshold(ctx)
#     recommendations.append({"mcs": mcs, "prob": prob})

# Real-time Integration:
# while True:
#     current_context = get_network_measurements()
#     recommended_mcs, confidence = recommender.recommend_mcs_threshold(current_context)
#     if confidence >= 0.8:  # High confidence threshold
#         apply_mcs_configuration(recommended_mcs)
    """)


def main():
    """Run the complete demonstration"""
    print("5G NTN MCS Recommendation System - Demo")
    print("="*60)
    
    try:
        # Initialize the recommender
        recommender = MCSRecommender()
        
        # Run demonstrations
        demonstrate_simple_inference(recommender)
        demonstrate_optimization(recommender)
        demonstrate_model_insights(recommender)
        demonstrate_practical_usage()
        
        print("\n" + "="*60)
        print("Demo completed successfully!")
        print("="*60)
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nTo resolve this:")
        print("1. Ensure you have trained the model: uv run python train_xgb.py --train features/train.parquet --test features/test.parquet")
        print("2. Or use the existing pipeline to create the required model files")
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise


if __name__ == "__main__":
    main()
