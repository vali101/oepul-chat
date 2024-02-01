import os
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup


# Manual other links:
# manual_links = [
#     "https://www.ama.at/getattachment/4d988952-847f-4847-a650-cf2e866d2ebe/20230131_Merkblatt_MFA2023_V3.pdf",
#     "https://www.ama.at/getattachment/abb20701-bbad-46cc-a5ef-deb2ab03ebc6/Merkblatt_Flaechenmonitoring_2023.pdf",
#     "https://www.ama.at/getattachment/05e60ad0-cbcb-40d3-820d-315e7fb49979/Nutzungsarten_Codes_Varianten_2023_V4.pdf"]


def download_data(data_export_path: str = "data/"):
    """
    Downloads data from OEPUL and Bio Austria websites.

    Args:
        data_export_path (str, optional): The path to export the downloaded data. Default: "data/".
    """
    # OEPUL
    links = __get_all_oepul_pdf_anchor_tags()
    __download_pdfs(
        links, data_export_path)

    # Bio Austria
    __download_html(
        url='https://www.bio-austria.at/a/bauern/aktueller-planungsstand-zu-bio-im-oepul-2023/',
        data_export_path=data_export_path,
        filename="aktueller-planungsstand-zu-bio-im-oepul-2023")


def __get_all_oepul_pdf_anchor_tags():
    # Initialize the web driver (make sure to specify the path to your webdriver)
    print("Retrieving links from OEPUL website...")
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
            link_div = driver.find_element(
                By.ID, "ui-id-21")
            div_present = True
        except Exception:
            pass

    # Parse the HTML content of the div using BeautifulSoup
    html = link_div.get_attribute(
        "innerHTML")
    soup = BeautifulSoup(
        html, 'html.parser')

    # Find and download all links in the div
    links = soup.find_all('a')
    links = [link.get('href') for link in links if link.get('href') is not None]

    driver.quit()

    return links


def __download_pdfs(links, data_export_path):
    url_base = "https://www.ama.at/"
    if not os.path.exists(f"{data_export_path}OEPUL_PDF"):
        os.makedirs(f"{data_export_path}OEPUL_PDF")

    for i, href in enumerate(links):
        print(f"Downloading OEPUL PDF {i+1}/{len(links)}", end="\r")
        # Make sure the link is an absolute URL
        # also check ig the link is a pdf
        if not href.startswith("http"):
            href = f"{url_base}{href}"
        if not href.endswith(".pdf"):
            continue

        # Download the PDF files (you can adjust the file path and handling as needed)
        response = requests.get(href, timeout=100)
        if response.status_code == 200:
            filename = href.split(
                "/")[-1]
            filepath = f"{data_export_path}OEPUL_PDF/{filename}"
            with open(filepath, 'wb') as f:
                f.write(
                    response.content)


def __download_html(url, data_export_path, filename):
    # Sending a GET request to the website
    print("Downloading Bio Austria HTML...")
    if not os.path.exists(f"{data_export_path}BIO_Austria"):
        os.makedirs(f"{data_export_path}BIO_Austria")

    response = requests.get(url, timeout=100)
    # Checking if the request was successful
    if response.status_code == 200:
        # save html to file
        with open(f'{data_export_path}BIO_Austria/{filename}.html', 'w', encoding='utf-8') as f:
            f.write(response.text)
