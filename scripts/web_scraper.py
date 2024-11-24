import requests
from bs4 import BeautifulSoup
import os

def get_last_image_number(folder_path, category):
    category_path = os.path.join(folder_path, category)
    if not os.path.exists(category_path):
        os.makedirs(category_path)
    files = os.listdir(category_path)
    if not files:
        return 0
    numbers = [int(f.split('_')[1].split('.')[0]) for f in files if f.startswith('image_')]
    return max(numbers) + 1 if numbers else 0

def download_images(url, folder_path, category):
    last_image_number = get_last_image_number(folder_path, category)
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    img_tags = soup.find_all('img')

    category_path = os.path.join(folder_path, category)
    if not os.path.exists(category_path):
        os.makedirs(category_path)

    for i, img in enumerate(img_tags):
        img_url = img.get('src')
        if img_url and img_url.startswith('http') and 'logo' not in img_url:
            img_response = requests.get(img_url)
            with open(os.path.join(category_path, f'image_{last_image_number + i}.jpg'), 'wb') as f:
                f.write(img_response.content)

# Existing URL
url_nude = 'paste/target/url'  # Replace with the target URL for nude images

# Download from the existing URL
download_images(url_nude, 'dataset', 'images')

# New URLs with loop
base_url = 'https://target/page-number'  # Base URL for new images
for i in range(1, 15):  # Adjust the range as needed
    new_url_nude = f'{base_url}{i}/'
    download_images(new_url_nude, 'dataset', 'images')
