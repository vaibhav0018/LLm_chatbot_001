import requests

# Send a POST request to the /refresh_data endpoint
response = requests.post('http://localhost:5000/refresh_data')

# Print the response from the server
print(response.json())