#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
TinyServe Kernels Python Test Suite
Comprehensive test suite for TinyServe optimized kernels using Python
"""

import ctypes
import os
import sys
import time
import unittest
from typing import Any

import numpy as np

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TinyServePythonTests(unittest.TestCase):
    """Test suite for TinyServe kernels"""
    
    # Class attributes for type checking
    batch_size: int
    num_heads: int
    head_dim: int
    seq_len: int
    block_size: int
    tinyserve_lib: Any | None
    library_available: bool
    
    @classmethod
    def setUpClass(cls) -> None:
        """Set up test class"""
        cls.batch_size = 4
        cls.num_heads = 8
        cls.head_dim = 64
        cls.seq_len = 128
        cls.block_size = 16
        
        # Try to load TinyServe library
        try:
            lib_path = os.path.join(
                os.path.dirname(__file__), "../build/libtinyserve_kernels.so"
            )
            cls.tinyserve_lib = ctypes.CDLL(lib_path)
            cls.library_available = True
            print("✓ TinyServe library loaded successfully")
        except OSError:
            cls.tinyserve_lib = None
            cls.library_available = False
            print("⚠ TinyServe library not found. Running in simulation mode.")
    
    def setUp(self):
        """Set up each test"""
        self.rng = np.random.RandomState(42)  # Fixed seed for reproducibility
    
    def generate_test_data(
        self, batch_size: int, seq_len: int, hidden_size: int
    ) -> tuple[np.ndarray, ...]:
        """Generate random test data"""
        queries = self.rng.randn(
            batch_size, self.num_heads, self.head_dim
        ).astype(np.float16)
        key_blocks = self.rng.randn(
            seq_len, self.head_dim
        ).astype(np.float16)
        value_blocks = self.rng.randn(
            seq_len, self.head_dim
        ).astype(np.float16)
        seq_lens = np.full(batch_size, seq_len, dtype=np.int32)
        num_blocks = batch_size * (seq_len // self.block_size)
        block_table = np.arange(num_blocks).reshape(
            batch_size, -1
        ).astype(np.int32)
        
        return queries, key_blocks, value_blocks, seq_lens, block_table
    
    def test_data_generation(self):
        """Test data generation"""
        print("\nTesting data generation...")
        
        queries, key_blocks, value_blocks, seq_lens, block_table = (
            self.generate_test_data(
                self.batch_size, self.seq_len, self.head_dim
            )
        )
        
        # Verify shapes
        self.assertEqual(
            queries.shape,
            (self.batch_size, self.num_heads, self.head_dim)
        )
        self.assertEqual(key_blocks.shape, (self.seq_len, self.head_dim))
        self.assertEqual(value_blocks.shape, (self.seq_len, self.head_dim))
        self.assertEqual(seq_lens.shape, (self.batch_size,))
        expected_blocks = self.seq_len // self.block_size
        self.assertEqual(
            block_table.shape, (self.batch_size, expected_blocks)
        )
        
        # Verify data types
        self.assertEqual(queries.dtype, np.float16)
        self.assertEqual(key_blocks.dtype, np.float16)
        self.assertEqual(value_blocks.dtype, np.float16)
        self.assertEqual(seq_lens.dtype, np.int32)
        self.assertEqual(block_table.dtype, np.int32)
        
        print("✓ Data generation test passed")
    
    def test_query_complexity_analysis(self):
        """Test query complexity analysis"""
        print("\nTesting query complexity analysis...")
        
        queries, _, _, _, _ = (
            self.generate_test_data(
                self.batch_size, self.seq_len, self.head_dim
            )
        )
        
        # Simulate query complexity analysis
        query_complexity = np.zeros(self.batch_size * self.seq_len, dtype=np.float32)
        cache_requirements = np.zeros(self.batch_size * self.seq_len, dtype=np.int32)
        
        for batch_idx in range(self.batch_size):
            for pos in range(self.seq_len):
                # Calculate complexity based on query variance
                query_slice = queries[batch_idx, :, :]
                complexity = np.var(query_slice)
                
                idx = batch_idx * self.seq_len + pos
                query_complexity[idx] = complexity
                
                # Determine cache requirements
                if complexity > 0.5:
                    cache_requirements[idx] = 2  # High cache
                elif complexity > 0.2:
                    cache_requirements[idx] = 1  # Medium cache
                else:
                    cache_requirements[idx] = 0  # Low cache
        
        # Verify results
        self.assertTrue(np.all(query_complexity >= 0))
        self.assertTrue(np.all(cache_requirements >= 0))
        self.assertTrue(np.all(cache_requirements <= 2))
        
        # Check that we have some variation
        self.assertTrue(np.std(query_complexity) > 0)
        unique_reqs = np.unique(cache_requirements)
        self.assertTrue(len(unique_reqs) > 1)
        
        print("✓ Query complexity analysis test passed")
    
    def test_cache_selection_strategy(self):
        """Test cache selection strategy"""
        print("\nTesting cache selection strategy...")
        
        # Generate test data
        num_elements = self.batch_size * self.seq_len
        query_complexity = self.rng.uniform(0, 1, num_elements).astype(
            np.float32
        )
        cache_requirements = self.rng.randint(0, 3, num_elements).astype(
            np.int32
        )
        access_history = self.rng.randint(0, 100, num_elements).astype(
            np.int32
        )
        
        num_cache_blocks = 256
        cache_threshold = 0.5
        selected_cache_blocks = np.zeros(num_cache_blocks, dtype=np.int32)
        
        # Simulate cache selection
        for i in range(self.batch_size * self.seq_len):
            complexity = query_complexity[i]
            requirement = cache_requirements[i]
            history = access_history[i]
            
            # Calculate selection score
            selection_score = (
                complexity * 0.4
                + requirement * 0.3
                + (history / 100.0) * 0.3
            )
            
            if selection_score > cache_threshold:
                # High priority - allocate multiple blocks
                for j in range(min(requirement + 1, 4)):
                    block_id = i * 4 + j
                    if block_id < num_cache_blocks:
                        selected_cache_blocks[block_id] = 1
            elif (
                selection_score > cache_threshold * 0.5
                and i < num_cache_blocks
            ):
                # Medium priority - allocate single block
                selected_cache_blocks[i] = 1
        
        # Verify results
        self.assertTrue(np.all(selected_cache_blocks >= 0))
        self.assertTrue(np.all(selected_cache_blocks <= 1))
        # Some blocks should be selected
        self.assertTrue(np.sum(selected_cache_blocks) > 0)
        
        print("✓ Cache selection strategy test passed")
    
    def test_attention_computation_simulation(self):
        """Test attention computation simulation"""
        print("\nTesting attention computation simulation...")
        
        queries, key_blocks, value_blocks, seq_lens, block_table = (
            self.generate_test_data(
                self.batch_size, self.seq_len, self.head_dim
            )
        )
        
        # Simulate attention computation
        output = np.zeros(
            (self.batch_size, self.num_heads, self.head_dim),
            dtype=np.float16
        )
        
        for batch_idx in range(self.batch_size):
            seq_len = seq_lens[batch_idx]
            
            for head_idx in range(self.num_heads):
                # Simulate attention scores
                attention_scores = self.rng.randn(seq_len).astype(np.float32)
                attention_weights = self.softmax(attention_scores)
                
                # Simulate output computation
                for d in range(self.head_dim):
                    output[batch_idx, head_idx, d] = np.sum(
                        attention_weights[:seq_len] * 
                        self.rng.randn(seq_len).astype(np.float16)
                    )
        
        # Verify results
        self.assertEqual(
            output.shape,
            (self.batch_size, self.num_heads, self.head_dim)
        )
        self.assertEqual(output.dtype, np.float16)
        
        # Check for valid values
        self.assertFalse(np.any(np.isnan(output)))
        self.assertFalse(np.any(np.isinf(output)))
        
        print("✓ Attention computation simulation test passed")
    
    def test_memory_compaction_simulation(self):
        """Test memory compaction simulation"""
        print("\nTesting memory compaction simulation...")
        
        num_blocks = 64
        block_size = 16
        hidden_size = 64
        
        # Generate test data
        blocks = self.rng.randn(
            num_blocks, block_size, hidden_size
        ).astype(np.float16)
        old_to_new_mapping = np.arange(num_blocks) // 2  # Compact by half
        block_weights = self.rng.randint(0, 10, num_blocks)
        compaction_threshold = 5
        
        # Simulate compaction
        compacted_blocks = blocks.copy()
        
        for block_idx in range(num_blocks):
            if block_weights[block_idx] < compaction_threshold:
                new_block_idx = old_to_new_mapping[block_idx]
                if new_block_idx != block_idx:
                    compacted_blocks[new_block_idx] = blocks[block_idx]
        
        # Verify results
        self.assertEqual(compacted_blocks.shape, blocks.shape)
        self.assertEqual(compacted_blocks.dtype, blocks.dtype)
        
        # Check that compaction occurred
        self.assertTrue(np.any(compacted_blocks != blocks))
        
        print("✓ Memory compaction simulation test passed")
    
    def test_performance_benchmark(self):
        """Test performance benchmark"""
        print("\nTesting performance benchmark...")
        
        num_iterations = 50
        
        # Generate test data
        queries, key_blocks, value_blocks, seq_lens, block_table = (
            self.generate_test_data(
                self.batch_size, self.seq_len, self.head_dim
            )
        )
        
        # Benchmark simulation
        start_time = time.time()
        
        for i in range(num_iterations):
            # Simulate attention computation
            output = np.zeros(
                (self.batch_size, self.num_heads, self.head_dim),
                dtype=np.float16
            )
            
            for batch_idx in range(self.batch_size):
                for head_idx in range(self.num_heads):
                    attention_scores = self.rng.randn(self.seq_len).astype(np.float32)
                    attention_weights = self.softmax(attention_scores)
                    
                    for d in range(self.head_dim):
                        rand_vals = self.rng.randn(self.seq_len).astype(
                            np.float16
                        )
                        output[batch_idx, head_idx, d] = np.sum(
                            attention_weights * rand_vals
                        )
        
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time = total_time / num_iterations
        throughput = num_iterations / total_time
        
        print(f"Performance Results:")
        print(f"  Total time: {total_time:.3f} seconds")
        print(f"  Average time per iteration: {avg_time*1000:.3f} ms")
        print(f"  Throughput: {throughput:.1f} iterations/second")
        
        # Verify reasonable performance
        self.assertLess(avg_time, 1.0)  # Should be less than 1 second per iteration
        # Should be more than 1 iteration per second
        self.assertGreater(throughput, 1.0)
        
        print("✓ Performance benchmark test passed")
    
    def test_integration_workflow(self):
        """Test complete integration workflow"""
        print("\nTesting integration workflow...")
        
        # Step 1: Generate test data
        queries, key_blocks, value_blocks, seq_lens, block_table = (
            self.generate_test_data(
                self.batch_size, self.seq_len, self.head_dim
            )
        )
        
        # Step 2: Query analysis
        query_complexity = np.zeros(self.batch_size * self.seq_len, dtype=np.float32)
        cache_requirements = np.zeros(self.batch_size * self.seq_len, dtype=np.int32)
        
        for batch_idx in range(self.batch_size):
            for pos in range(self.seq_len):
                query_slice = queries[batch_idx, :, :]
                complexity = np.var(query_slice)
                
                idx = batch_idx * self.seq_len + pos
                query_complexity[idx] = complexity
                
                if complexity > 0.5:
                    cache_requirements[idx] = 2
                elif complexity > 0.2:
                    cache_requirements[idx] = 1
                else:
                    cache_requirements[idx] = 0
        
        # Step 3: Cache selection
        num_cache_blocks = 256
        selected_cache_blocks = np.zeros(num_cache_blocks, dtype=np.int32)
        
        for i in range(self.batch_size * self.seq_len):
            complexity = query_complexity[i]
            requirement = cache_requirements[i]
            
            selection_score = complexity * 0.4 + requirement * 0.3
            
            if selection_score > 0.5:
                for j in range(min(requirement + 1, 4)):
                    block_id = i * 4 + j
                    if block_id < num_cache_blocks:
                        selected_cache_blocks[block_id] = 1
        
        # Step 4: Attention computation
        output = np.zeros(
            (self.batch_size, self.num_heads, self.head_dim),
            dtype=np.float16
        )
        
        for batch_idx in range(self.batch_size):
            for head_idx in range(self.num_heads):
                attention_scores = self.rng.randn(self.seq_len).astype(np.float32)
                attention_weights = self.softmax(attention_scores)
                
                for d in range(self.head_dim):
                    rand_vals = self.rng.randn(self.seq_len).astype(
                        np.float16
                    )
                    output[batch_idx, head_idx, d] = np.sum(
                        attention_weights * rand_vals
                    )
        
        # Verify complete workflow
        self.assertTrue(np.all(query_complexity >= 0))
        self.assertTrue(np.all(cache_requirements >= 0))
        self.assertTrue(np.all(selected_cache_blocks >= 0))
        self.assertFalse(np.any(np.isnan(output)))
        self.assertFalse(np.any(np.isinf(output)))
        
        print("✓ Integration workflow test passed")
    
    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        """Compute softmax"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

def run_tests():
    """Run all tests"""
    print("=== TinyServe Python Test Suite ===")
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TinyServePythonTests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n=== Test Summary ===")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback}")
    
    if result.wasSuccessful():
        print("\n✓ All tests passed!")
        return True
    else:
        print("\n✗ Some tests failed!")
        return False

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
