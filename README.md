# Crypto News Scraper and Analyzer

This application scrapes cryptocurrency market data from [Polymarket](https://polymarket.com/markets/crypto), analyzes relevant news using an LLM, and outputs the results in JSON format for human curation.

## Features

- **Web Scraping:** Extracts market data from Polymarket using Selenium.
- **LLM Analysis:** Sends scraped data to an LLM for analysis with a system prompt.
- **JSON Output:** Outputs news in JSON format with a `relevancy` boolean for review.

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/crypto-news-scraper.git
cd crypto-news-scraper
```

### 2. Create a Virtual Environment and Install Dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure Environment Variables

- Rename `env.example` to `.env`:
  
  ```bash
  mv env.example .env
  ```

- Open `.env` and set your `GLHF_API_KEY`:

  ```
  GLHF_API_KEY=your_glhf_api_key_here
  ```

### 4. Run the Application

```bash
python app/main.py
```

### 5. Testing

Run the tests to ensure everything is working correctly:

```bash
pytest
```

## Project Structure

- `app/main.py`: Main application script.
- `templates/`: Contains template documentation.
- `tests/test_main.py`: Contains unit tests.
- `.gitignore`: Specifies intentionally untracked files.
- `env.example`: Example environment variables.
- `README.md`: Project documentation.
- `requirements.txt`: Python dependencies.

## Dependencies

- **Selenium:** For web scraping.
- **webdriver-manager:** To manage WebDriver binaries.
- **OpenAI:** For interacting with the LLM.
- **python-dotenv:** To manage environment variables.
- **pytest:** For testing.

## License

MIT License