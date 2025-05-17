from script import predict  # Make sure to update script.py with the fixed predict function


result = predict('./data/bid_requests_test.csv', './data/expected_results.csv')

print("File 'expected_results.csv' generated successfully!")