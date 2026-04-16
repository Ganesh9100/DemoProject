import org.openqa.selenium.By;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.WebElement;
import org.openqa.selenium.chrome.ChromeDriver;

public class Main {
    public static void main(String[] args) {
        // The platform usually handles the driver path
        WebDriver driver = new ChromeDriver();

        try {
            driver.get("https://www.google.com");
            
            // Locate and click the button
            WebElement button = driver.findElement(By.id("submit-btn"));
            button.click();

        } catch (Exception e) {
            System.out.println("Error: " + e.getMessage());
        } finally {
            driver.quit();
        }
    }
}
