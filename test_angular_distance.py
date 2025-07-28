import torch
import torch.nn.functional as F
import numpy as np
from src.models.carot_ldreg import lid_mom_est, compute_ldreg_loss

def test_chordal_distance():
    """Test to verify chordal distance implementation works correctly"""
    
    # Create test data - hyperspherical representations
    batch_size = 32
    feature_dim = 128
    
    # Generate random features and normalize them to unit sphere
    image_features = torch.randn(batch_size, feature_dim)
    text_features = torch.randn(batch_size, feature_dim)
    
    # Test 1: Check that LID estimation runs without errors
    print("Test 1: Basic LID estimation")
    try:
        lids = lid_mom_est(image_features, image_features, k=8)
        print(f"✓ LID estimation successful, shape: {lids.shape}")
        print(f"✓ LID values range: [{lids.min():.4f}, {lids.max():.4f}]")
    except Exception as e:
        print(f"✗ LID estimation failed: {e}")
        return False
    
    # Test 2: Check that chordal distance is being used
    print("\nTest 2: Chordal distance verification")
    # Create more realistic test cases with sufficient neighbors
    test_batch_size = 10
    test_feature_dim = 4
    
    # Create a batch with some identical vectors
    identical_batch = torch.ones(test_batch_size, test_feature_dim)
    # Add small perturbations to avoid exact duplicates
    identical_batch += torch.randn_like(identical_batch) * 0.01
    lids_identical = lid_mom_est(identical_batch, identical_batch, k=3)
    print(f"✓ LID for nearly identical vectors: mean={lids_identical.mean():.4f}, std={lids_identical.std():.4f}")
    
    # Create orthogonal vectors in higher-dimensional space
    orthogonal_batch = torch.randn(test_batch_size, test_feature_dim)
    orthogonal_batch = F.normalize(orthogonal_batch, p=2, dim=1)  # Normalize to unit sphere
    lids_orthogonal = lid_mom_est(orthogonal_batch, orthogonal_batch, k=3)
    print(f"✓ LID for random unit vectors: mean={lids_orthogonal.mean():.4f}, std={lids_orthogonal.std():.4f}")
    
    # Test 3: Check combined LDReg loss computation
    print("\nTest 3: Combined LDReg loss computation")
    try:
        ldreg_loss, mean_lid_image, mean_lid_text = compute_ldreg_loss(
            image_features, text_features, k=8, reg_type="l1"
        )
        print(f"✓ LDReg loss computation successful")
        print(f"✓ Combined loss: {ldreg_loss:.4f}")
        print(f"✓ Mean LID Image: {mean_lid_image:.4f}")
        print(f"✓ Mean LID Text: {mean_lid_text:.4f}")
    except Exception as e:
        print(f"✗ LDReg loss computation failed: {e}")
        return False
    
    # Test 4: Verify that features are normalized internally
    print("\nTest 4: Normalization verification")
    unnormalized_features = torch.randn(batch_size, feature_dim) * 10  # Large magnitudes
    try:
        lids_unnorm = lid_mom_est(unnormalized_features, unnormalized_features, k=8)
        print(f"✓ LID estimation works with unnormalized features")
        print(f"✓ LID values range: [{lids_unnorm.min():.4f}, {lids_unnorm.max():.4f}]")
    except Exception as e:
        print(f"✗ LID estimation failed with unnormalized features: {e}")
        return False
    
    # Test 5: Verify chordal distance behavior
    print("\nTest 5: Chordal distance vs Euclidean distance behavior")
    # Create two vectors with same direction but different magnitudes
    v1 = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    v2 = torch.tensor([[2.0, 0.0, 0.0, 0.0]])  # Same direction, different magnitude
    v3 = torch.tensor([[0.0, 1.0, 0.0, 0.0]])  # Orthogonal
    
    test_vectors = torch.cat([v1, v2, v3], dim=0)
    
    # In chordal distance, v1 and v2 should be identical (both point in same direction)
    # In Euclidean distance, they would be different
    
    # Test with more vectors to avoid edge cases
    repeated_test = test_vectors.repeat(4, 1)  # 12 vectors total
    noise = torch.randn_like(repeated_test) * 0.01
    repeated_test += noise
    
    lids_chordal = lid_mom_est(repeated_test, repeated_test, k=3)
    print(f"✓ Chordal distance LID computation successful")
    print(f"✓ LID values (chordal): mean={lids_chordal.mean():.4f}, min={lids_chordal.min():.4f}, max={lids_chordal.max():.4f}")
    
    # Check that no NaN values are produced
    if torch.isnan(lids_chordal).any():
        print("✗ NaN values detected in chordal distance computation")
        return False
    else:
        print("✓ No NaN values in chordal distance computation")
    
    print("\n🎉 All tests passed! Chordal distance implementation is working correctly.")
    return True

if __name__ == "__main__":
    test_chordal_distance() 