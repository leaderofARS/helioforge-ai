import sys
import torch
import traceback
from pathlib import Path

# Add repository root directory to Python path
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

def test_causal_conv():
    print("Testing causal_conv.py...")
    from src.HPINA.models.baseline_tcn.causal_conv import CausalConv1d
    x = torch.randn(2, 32, 512)
    # Test different kernel sizes and dilations
    for k in [3, 5]:
        for d in [1, 2, 4]:
            conv = CausalConv1d(in_channels=32, out_channels=64, kernel_size=k, dilation=d)
            out = conv(x)
            assert out.shape == (2, 64, 512), f"Failed for k={k}, d={d}. Got shape {out.shape}"
    print("✓ causal_conv.py tests passed successfully!\n")

def test_residual_block():
    print("Testing residual_block.py...")
    from src.HPINA.models.baseline_tcn.residual_block import TemporalResidualBlock, LayerNorm1d
    
    # Test LayerNorm1d wrapper
    x = torch.randn(2, 32, 512)
    ln = LayerNorm1d(32)
    assert ln(x).shape == (2, 32, 512)
    
    # Test TemporalResidualBlock with batch norm, layer norm, and none
    for norm in ["batch", "layer", "none"]:
        for d in [1, 4, 16]:
            block = TemporalResidualBlock(in_channels=32, out_channels=64, kernel_size=3, dilation=d, norm_type=norm)
            out = block(x)
            assert out.shape == (2, 64, 512), f"Failed for norm={norm}, d={d}. Got shape {out.shape}"
    print("✓ residual_block.py tests passed successfully!\n")

def test_tcn_encoder():
    print("Testing tcn_encoder.py...")
    from src.HPINA.models.baseline_tcn.tcn_encoder import TCNEncoder
    x = torch.randn(2, 32, 512)
    
    # Test default progressive widening schedule
    encoder = TCNEncoder(in_channels=32)
    assert encoder.out_channels == 512
    out = encoder(x)
    assert out.shape == (2, 512, 512)
    
    # Test custom schedule
    encoder_custom = TCNEncoder(in_channels=32, channel_schedule=[64, 128])
    assert encoder_custom.out_channels == 128
    out_custom = encoder_custom(x)
    assert out_custom.shape == (2, 128, 512)
    print("✓ tcn_encoder.py tests passed successfully!\n")

def test_classifier():
    print("Checking classifier.py...")
    try:
        from src.HPINA.models.baseline_tcn.classifier import ClassifierHead
        # Test ClassifierHead if defined
        head = ClassifierHead(in_features=512, n_classes=5)
        x = torch.randn(2, 512, 128)  # 3D tensor: (Batch, in_features, sequence_length)
        out = head(x)
        assert out.shape == (2, 5), f"Expected shape (2, 5) but got {out.shape}"
        print("✓ classifier.py tests passed successfully!\n")

    except ImportError:
        print("⚠ classifier.py does not contain ClassifierHead yet (skipping).\n")
    except AttributeError:
        print("⚠ ClassifierHead attribute not found in classifier.py (skipping).\n")
    except Exception as e:
        print(f"✗ classifier.py test failed: {e}")
        traceback.print_exc()
        print()

def test_model():
    print("Checking model.py...")
    try:
        from src.HPINA.models.baseline_tcn.model import HelioForgeTCN
        # Test HelioForgeTCN if defined
        model = HelioForgeTCN(in_channels=32, n_classes=5)
        x = torch.randn(2, 32, 512)
        out = model(x)
        assert out.shape == (2, 5), f"Expected shape (2, 5) but got {out.shape}"
        print("✓ model.py tests passed successfully!\n")
    except ImportError:
        print("⚠ model.py does not contain HelioForgeTCN yet (skipping).\n")
    except AttributeError:
        print("⚠ HelioForgeTCN attribute not found in model.py (skipping).\n")
    except Exception as e:
        print(f"✗ model.py test failed: {e}")
        traceback.print_exc()
        print()

def main():
    print("=" * 60)
    print("Running Baseline TCN Component Tests")
    print("=" * 60)
    
    tests = [
        test_causal_conv,
        test_residual_block,
        test_tcn_encoder,
        test_classifier,
        test_model
    ]
    
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"✗ Critical failure in {test.__name__}: {e}")
            traceback.print_exc()
            print()
            
    print("=" * 60)
    print("Tests execution finished.")
    print("=" * 60)

if __name__ == "__main__":
    main()
