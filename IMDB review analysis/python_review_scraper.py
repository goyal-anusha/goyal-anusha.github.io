import requests
import csv
import tqdm
import json
import aiohttp
import asyncio

import tqdm.asyncio


# Function to fetch reviews from IMDb Movie Scraper API
def fetch_reviews(movie_id):
    api_url = f'https://imdb-rest-api.herokuapp.com/api/livescraper/reviews/{movie_id}'
    response = requests.get(api_url)
    
    if response.status_code == 200:
        return response.json().get('reviews', [])
    else:
        return None

# Function to fetch movie IDs from your table
def fetch_movie_ids_from_csv():
    csv_file = "/Users/anusha/Desktop/Columbia School/frameworks2/Project/clean_dataset.csv"
    # Read the movie IDs from the CSV file
    with open(csv_file, 'r', encoding="utf-8") as file:
        reader = csv.reader(file)
        # Assuming the 'id' column is in the first column of the CSV
        movie_ids = [row[0] for row in reader]
    return list(set(movie_ids))

def export_reviews_to_csv(reviews_data, csv_file):
    try:
        with open(csv_file, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)

            # Write header
            writer.writerow(['id', 'review'])

            # Write reviews data
            for data in reviews_data:
                movie_id = data['id']
                reviews = data['reviews']
                for review in reviews:
                    writer.writerow([movie_id, review])
        print("Reviews exported to CSV successfully.")
    except Exception as e:
        print("Error exporting reviews to CSV:", e)

reviews_data = []

# Running the script takes a while so we will be running it on a select few for the proposal to generate an output
# movie_ids = ["tt0102926", "tt0138704", "tt0468569", "tt0988130", "tt10680412"]

movie_ids = fetch_movie_ids_from_csv()

movie_id_to_url = {}
for movie_id in movie_ids:
    movie_id_to_url[movie_id] = f'https://imdb-rest-api.herokuapp.com/api/livescraper/reviews/{movie_id}'

conn = aiohttp.TCPConnector(limit_per_host=100, limit=0, ttl_dns_cache=10)
PARALLEL_REQUESTS = 100

print("starting async operations now")


async def gather_with_concurrency(n):
    semaphore = asyncio.Semaphore(n)
    session = aiohttp.ClientSession(connector=conn)
    
    async def get(movie_id, url):
        async with semaphore:
            async with session.get(url, ssl=False) as response:
                obj = json.loads(await response.read())
                
                reviews_data.append({'id': movie_id, 'reviews': obj["reviews"]})
    await tqdm.asyncio.tqdm.gather(*(get(movie_id, url) for movie_id, url in movie_id_to_url.items()))
    
    await session.close()

loop = asyncio.get_event_loop()
loop.run_until_complete(gather_with_concurrency(PARALLEL_REQUESTS))
conn.close()


# Export reviews to CSV
export_reviews_to_csv(reviews_data, '/Users/anusha/Desktop/Columbia School/frameworks2/Project/reviews_output.csv')

print("Scraping and storing reviews completed.")

print(reviews_data)
