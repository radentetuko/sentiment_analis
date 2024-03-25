from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import pandas as pd

url = input("Masukkan URL toko: ")

def scrape_reviews(url):
    if url:
        options = webdriver.ChromeOptions()
        options.add_argument("--start-maximized")
        driver = webdriver.Chrome(options=options)
        driver.get(url)

        time.sleep(2)

        # Scroll ke bawah dan mengklik ulasan
        driver.execute_script("window.scrollTo(0, window.scrollY + 300);")
        WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.CSS_SELECTOR, "button[data-testid='review']"))).click()
        
        data = []
        for i in range(0, 10):  # Ganti angka 10 menjadi 100 untuk mendapatkan 100 data
            soup = BeautifulSoup(driver.page_source, "html.parser")
            containers = soup.findAll('article', attrs={'class': 'css-72zbc4'})

            for container in containers:
                try:
                    review = container.find('span', attrs={'data-testid': 'lblItemUlasan'}).text
                    rating = container.find('div', attrs={'data-testid': 'icnStarRating'}).get('aria-label')
                    timestamp = container.find('time').get('datetime')  # Ambil waktu ulasan
                    data.append({"Review": review, "Rating": rating, "Timestamp": timestamp})
                except AttributeError:
                    continue

            time.sleep(2)
            try:
                WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.CSS_SELECTOR, "button[aria-label^='Laman berikutnya']"))).click()
            except:
                break
            time.sleep(3)

        driver.quit()

        # Membuat DataFrame dari data
        df = pd.DataFrame(data)

        # Menyimpan DataFrame ke dalam file CSV
        # Menambahkan pengecekan kondisi
        if not df.empty:
            df.to_csv('./reviews.csv', index=False)
            print(data)
            print(df)
            print("Data telah disimpan ke dalam file reviews.csv")
        else:
            print("Data tidak ditemukan atau tidak valid.")

scrape_reviews(url)


