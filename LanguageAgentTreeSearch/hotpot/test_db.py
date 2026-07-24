import os
from wikienv import WikiEnv
import time

def parse_test_file(filepath):
    """Parses the text file into a list of (entity, expected_observation) tuples."""
    if not os.path.exists(filepath):
        print(f"Error: Could not find {filepath}")
        return []

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split the file by the "Search Entity: " marker
    blocks = content.split('Search Entity: ')
    tests = []
    
    for block in blocks:
        if not block.strip():
            continue
        
        # Split each block into the entity and the expected result
        parts = block.split('\nObservation / Wiki Result:\n')
        if len(parts) == 2:
            entity = parts[0].strip()
            expected = parts[1].strip()
            
            # Skip empty trailing blocks (like the final Heisman winner block in your example)
            if expected:
                tests.append((entity, expected))
            
    return tests

def run_tests():
    test_file = "/mnt/scratch/scheduler/proxy_server/wiki900_905.txt"
    tests = parse_test_file(test_file)
    
    if not tests:
        print("No tests found to run.")
        return

    print(f"Found {len(tests)} tests. Initializing local database environment...\n")
    
    # Initialize your local Wikipedia environment
    env = WikiEnv()
    env.reset()
    
    passed_count = 0
    failed_count = 0
    start_time = time.time()
    for i, (entity, expected_obs) in enumerate(tests, 1):
        print(f"Test {i}/{len(tests)}: Searching for '{entity}'...")
        
        # Trigger the search step in your environment
        obs, reward, done, info = env.step(f"search[{entity}]")
        
        # Clean up strings for comparison
        actual_obs = str(obs).strip()
        expected_obs = str(expected_obs).strip()
        
        if actual_obs == expected_obs:
            print("✅ PASS")
            passed_count += 1
        else:
            print("❌ FAIL")
            print(f"\n--- EXPECTED ---\n{expected_obs[:200]}...")
            print(f"\n--- ACTUAL ---\n{actual_obs[:200]}...")
        print("-" * 60)

    # Clean up the database connection
    env.close()

    # Print final summary
    print("\n" + "=" * 30)
    print("TESTING SUMMARY")
    print(f"Total Time: {time.time() - start_time:.10f} seconds")
    print("=" * 30)
    print(f"Total Tests: {len(tests)}")
    print(f"Passed:      {passed_count}")
    print(f"Failed:      {failed_count}")
    print("=" * 30)

if __name__ == "__main__":
    run_tests()