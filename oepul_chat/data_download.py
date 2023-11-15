from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup
import requests


def get_all_oepul_pdf_anchor_tags():
    # Initialize the web driver (make sure to specify the path to your webdriver)
    driver = webdriver.Firefox()

    # Open the website
    url = "https://www.ama.at/formulare-merkblaetter#18053"
    driver.get(url)

    div_present = False
    link_div = None
    while not div_present:
        driver.implicitly_wait(1)
        try:
            # Find the div with id "ui-id-21"
            link_div = driver.find_element(By.ID, "ui-id-21")
            div_present = True
        except Exception:
            pass

    # Parse the HTML content of the div using BeautifulSoup
    html = link_div.get_attribute("innerHTML")
    soup = BeautifulSoup(html, 'html.parser')

    # Find and download all links in the div
    links = soup.find_all('a')

    driver.quit()

    return links


def download_pdfs(links):
    url_base = "https://www.ama.at/"
    for link in links:
        href = link.get('href')
        # Make sure the link is an absolute URL
        # also check ig the link is a pdf
        if not href.startswith("http"):
            href = f"{url_base}{href}"
        if not href.endswith(".pdf"):
            continue

        # Download the PDF files (you can adjust the file path and handling as needed)
        response = requests.get(href)
        if response.status_code == 200:
            filename = href.split("/")[-1]
            filepath = f"data/OEPUL_PDF/{filename}"
            with open(filepath, 'wb') as f:
                f.write(response.content)


def download_html(url, folder_path, filename):
    # Sending a GET request to the website
    response = requests.get(url)
    # Checking if the request was successful
    if response.status_code == 200:
        # save html to file
        with open(f'{folder_path}/{filename}.html', 'w') as f:
            f.write(response.text)
