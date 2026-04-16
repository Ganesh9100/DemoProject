from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

def run_test():
    # 1. Setup the Chrome driver
    # Using ChromeDriverManager automatically handles the driver executable
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))

    try:
        # 2. Navigate to the webpage
        driver.get("https://www.example.com") # Replace with your target URL
        
        # 3. Maximize window (optional but recommended)
        driver.maximize_window()

        # 4. Find the button by ID and click it
        # The prompt specifically asked for ID 'submit-btn'
        submit_button = driver.find_element(By.ID, "submit-btn")
        submit_button.click()

        print("Successfully clicked the button.")

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        # 5. Close the browser session
        driver.quit()

if __name__ == "__main__":
    run_test()
