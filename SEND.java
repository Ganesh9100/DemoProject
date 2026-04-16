import org.openqa.selenium.By;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.WebElement;
import org.openqa.selenium.chrome.ChromeDriver;

public class AutomationInterviewTask {
    public static void main(String[] args) {
        // Set the path to your chromedriver executable
        // System.setProperty("webdriver.chrome.driver", "path/to/chromedriver");

        // 1. Initialize the WebDriver
        WebDriver driver = new ChromeDriver();

        try {
            // 2. Navigate to the webpage
            driver.get("https://www.google.com"); // Replace with your target URL
            
            // 3. Locate the button using the ID 'submit-btn'
            WebElement button = driver.findElement(By.id("submit-btn"));

            // 4. Interact with the element
            button.click();

            System.out.println("Navigation and click action performed successfully.");

        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            // 5. Close the browser
            driver.quit();
        }
    }
}
