from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import requests
import re
from geopy.geocoders import Nominatim
import logging
from tqdm import tqdm
import numpy as np

logger = logging.getLogger(__name__)

def find_cbg(shp, points):
    cbgs = []
    for i in tqdm(points):
        cbgs.append(
            list(shp.loc[shp.geometry.contains(i), "cd_setor"].values)
        )
    cbgs = [i[0] if i != [] else np.nan for i in cbgs]
    #cbgs = list(chain.from_iterable(cbgs))
    return cbgs

def find_regex_cep(text: str) -> str:
    padrao = r"\d{5}-\d{3}"
    resultado = re.search(padrao,
                          text)
    if resultado:
        return resultado.group()
    else:
        return None
    
def search_cep(endereco):
    geolocator = Nominatim(
        user_agent = "my_geocoder"
    )
    location = geolocator.geocode(
        endereco,
        exactly_one = False
    )
    if location is not None:
        cep = find_regex_cep(location[0].raw['display_name'])
        return location, cep
    else:
        return location, None   
    
def find_census_area_by_zip(zip_code: str) -> str:
    url = f'https://viacep.com.br/ws/{zip_code}/json/'
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        census_area = data.get('ibge')
        return str(census_area)
    else: 
        return -1
    
def get_connection(url: str,
                   root_xpath: str) -> tuple:
    # Configurar o Selenium para executar o Chrome em modo headless
    chrome_options = Options()
    chrome_options.add_argument('--headless')  # Executar o Chrome em modo headless
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')

    driver = webdriver.Chrome(options=chrome_options)
    driver.get(url)
    limit_find, k = False, 0
    list_local = driver.find_elements(By.XPATH, root_xpath)
    print('- number of locals found: [{}]'.format(len(list_local)))
    print()
    while len(list_local)==0:
        print(f'trying: [{k + 1}]')
        driver.get(url)
        list_local = driver.find_elements(By.XPATH, root_xpath)
        if k >= 5:
            limit_find = True
            break
        k += 1
    return driver, list_local, limit_find

def scrapping_zipimoveis(url: str) -> dict:
    root_xpath = "//div[@class='card-container js-listing-card']"
    i_xpath_highlight = root_xpath + "//div[@class='simple-card__highligths']"
    i_xpath_price = root_xpath + "//div[@class='simple-card__listing-prices simple-card__prices']"
    i_xpath_amenities = root_xpath + "//ul[@class='feature__container simple-card__amenities']"
    i_xpath_feature = ".//li[contains(@class, 'feature__item')]"
    i_xpath_local = ".//h2[@class='simple-card__address color-dark text-regular']"
    try:

        driver, list_local, _ = get_connection(
            url, root_xpath
        )

        content_data = {}
        for idx, i in enumerate(list_local):
            ID = f'id-{i.get_attribute("data-id")}'
            content_data[ID] = {}
            
            #print('-- getting: [{}]'.format(ID))
            price = i.find_elements(By.XPATH,
                                    i_xpath_price)
            price_extracted = price[idx].find_element(By.TAG_NAME, "strong").text
        
            
            highlight = i.find_elements(By.XPATH,
                                        i_xpath_highlight)
            try:
                highlight_extracted = highlight[idx].find_element(By.TAG_NAME, "strong").text
            except:
                highlight_extracted = np.nan
            content_data[ID]['highlight'] = [
                highlight_extracted
            ]
            #print('---- price: [{}]'.format(price_extracted))
            content_data[ID]['price'] = [
                price_extracted
            ]

            local = i.find_element(By.XPATH,
                                   i_xpath_local)

            local_extracted = local.text
            content_data[ID]['local'] = [local_extracted]
            #print('---- local: [{}]'.format(local_extracted))

            card_amenities = i.find_elements(By.XPATH, 
                                             i_xpath_amenities)

            elementos = card_amenities[idx].find_elements(By.XPATH,
                                                          i_xpath_feature)
            content_data[ID]['features'] = {
                        i.get_attribute("class").split(" ")[-1][3:]: i.text.strip() for i in elementos
                    }
            #print()
    except Exception as e:
        print('Erro durante a execução:', e)

    finally:
        driver.quit()
    return content_data