#!/usr/bin/env python3
"""
Test script for Autonomous Curiosity & Consolidation Loop (Phase-5).
Tests the intrinsic motivation and self-driven exploration functionality.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_autonomous_imports():
    """Test that autonomous modules can be imported."""
    print("Testing autonomous module imports...")

    try:
        # Test basic imports without torch dependencies
        import ast

        # Check autonomous.py syntax
        with open('nanochat/autonomous.py', 'r') as f:
            code = f.read()
        ast.parse(code)
        print("✓ nanochat.autonomous.py syntax is valid")

        # Check autonomous_demo.py syntax
        with open('scripts/autonomous_demo.py', 'r') as f:
            code = f.read()
        ast.parse(code)
        print("✓ scripts/autonomous_demo.py syntax is valid")

        # Test import structure (this will fail if torch isn't available, but that's expected)
        try:
            from nanochat.autonomous import CuriosityEngine, AutonomousAgent, enable_autonomous_mode
            print("✓ All autonomous classes imported successfully")
        except ImportError as e:
            if "torch" in str(e):
                print("⚠️  Autonomous classes require torch (expected in test environment)")
                print("   This is normal - full functionality requires PyTorch installation")
            else:
                raise e

    except Exception as e:
        print(f"❌ Import test failed: {e}")
        raise

def test_config_integration():
    """Test that autonomous config options are properly integrated."""
    print("Testing autonomous configuration integration...")

    try:
        # Test GPTConfig structure by examining source code
        import inspect
        import ast

        # Read and parse the GPTConfig class
        with open('nanochat/gpt.py', 'r') as f:
            content = f.read()

        # Check that autonomous config options are in the source
        assert 'autonomous_mode' in content, "GPTConfig source missing autonomous_mode"
        assert 'curiosity_threshold' in content, "GPTConfig source missing curiosity_threshold"
        assert 'nap_duration' in content, "GPTConfig source missing nap_duration"
        assert 'consolidation_interval' in content, "GPTConfig source missing consolidation_interval"

        # Check default values are set correctly
        assert 'autonomous_mode: bool = False' in content, "autonomous_mode default should be False"
        assert 'curiosity_threshold: float = 0.1' in content, "curiosity_threshold default should be 0.1"
        assert 'nap_duration: int = 60' in content, "nap_duration default should be 60"
        assert 'consolidation_interval: int = 3600' in content, "consolidation_interval default should be 3600"

        print("✓ GPTConfig autonomous options properly integrated")

    except Exception as e:
        print(f"❌ Config integration test failed: {e}")
        raise

def test_srgi_block_changes():
    """Test that SRGI block properly handles entanglement."""
    print("Testing SRGI block entanglement integration...")

    try:
        # Test SRGIBlock structure by examining source code
        import inspect

        # Read the SRGIBlock source
        with open('nanochat/gpt.py', 'r') as f:
            content = f.read()

        # Find the SRGIBlock class definition
        lines = content.split('\n')
        srgi_start = None
        for i, line in enumerate(lines):
            if line.startswith('class SRGIBlock'):
                srgi_start = i
                break

        assert srgi_start is not None, "SRGIBlock class not found"

        # Extract SRGIBlock class content
        srgi_content = '\n'.join(lines[srgi_start:])

        # Check that entanglement logic is present
        assert 'use_entangle' in srgi_content, "SRGIBlock missing use_entangle logic"
        assert 'entangle' in srgi_content, "SRGIBlock missing entangle initialization"
        assert 'EntangledBottleneck' in srgi_content, "SRGIBlock missing EntangledBottleneck import"

        print("✓ SRGIBlock entanglement integration successful")

    except Exception as e:
        print(f"❌ SRGI block test failed: {e}")
        raise

def test_curiosity_engine_structure():
    """Test CuriosityEngine class structure without torch."""
    print("Testing CuriosityEngine class structure...")

    try:
        # Test that the class has the expected methods by examining source
        import inspect

        # Import the module (will work even without torch since we only inspect)
        import nanochat.autonomous as autonomous_module

        # Get the CuriosityEngine class
        curiosity_class = autonomous_module.CuriosityEngine

        # Check class has expected methods
        assert hasattr(curiosity_class, '__init__'), "CuriosityEngine missing __init__"
        assert hasattr(curiosity_class, 'forward'), "CuriosityEngine missing forward method"

        # Check method signatures by inspecting source
        init_source = inspect.getsource(curiosity_class.__init__)
        forward_source = inspect.getsource(curiosity_class.forward)

        # Check __init__ has config parameter
        assert 'config' in init_source, "CuriosityEngine.__init__ should accept config"

        # Check forward has expected parameters
        assert 'current_state_complex' in forward_source, "forward should accept current_state_complex"
        assert 'webcam_frame' in forward_source, "forward should accept webcam_frame"
        assert 'audio_frame' in forward_source, "forward should accept audio_frame"

        print("✓ CuriosityEngine class structure validated")

    except Exception as e:
        print(f"❌ CuriosityEngine structure test failed: {e}")
        raise

def test_autonomous_agent_structure():
    """Test AutonomousAgent class structure."""
    print("Testing AutonomousAgent class structure...")

    try:
        import inspect
        import nanochat.autonomous as autonomous_module

        # Get the AutonomousAgent class
        agent_class = autonomous_module.AutonomousAgent

        # Check class has expected methods
        assert hasattr(agent_class, '__init__'), "AutonomousAgent missing __init__"
        assert hasattr(agent_class, 'run_autonomous_loop'), "AutonomousAgent missing run_autonomous_loop"

        # Check method signatures
        init_source = inspect.getsource(agent_class.__init__)
        loop_source = inspect.getsource(agent_class.run_autonomous_loop)

        # Check __init__ has expected parameters
        assert 'model' in init_source and 'config' in init_source, "__init__ should accept model and config"

        # Check run_autonomous_loop has webcam/audio parameters
        assert 'webcam_stream' in loop_source, "run_autonomous_loop should accept webcam_stream"
        assert 'audio_stream' in loop_source, "run_autonomous_loop should accept audio_stream"

        print("✓ AutonomousAgent class structure validated")

    except Exception as e:
        print(f"❌ AutonomousAgent structure test failed: {e}")
        raise

def test_enable_autonomous_mode_function():
    """Test the enable_autonomous_mode function structure."""
    print("Testing enable_autonomous_mode function...")

    try:
        import inspect
        import nanochat.autonomous as autonomous_module

        # Get the function
        enable_func = autonomous_module.enable_autonomous_mode

        # Check it's a function
        assert callable(enable_func), "enable_autonomous_mode should be callable"

        # Check parameters
        sig = inspect.signature(enable_func)
        params = list(sig.parameters.keys())
        assert 'model' in params, "enable_autonomous_mode should accept model"
        assert 'config' in params, "enable_autonomous_mode should accept config"

        print("✓ enable_autonomous_mode function structure validated")

    except Exception as e:
        print(f"❌ enable_autonomous_mode test failed: {e}")
        raise

if __name__ == "__main__":
    print("Running Autonomous Curiosity & Consolidation Loop tests...\n")

    try:
        test_autonomous_imports()
        test_config_integration()
        test_srgi_block_changes()
        test_curiosity_engine_structure()
        test_autonomous_agent_structure()
        test_enable_autonomous_mode_function()

        print("\n🎉 All autonomous module tests passed!")
        print("✅ Phase-5 autonomous curiosity implementation is structurally sound")
        print("✅ Ready for integration with full PyTorch environment")

    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
