# Portfolio Diversification App

The **Portfolio Diversification App** is a Streamlit-based application that helps investors and analysts explore portfolio diversification using both classical and quantum machine learning techniques. The app downloads historical stock data for S&P 500 companies, computes performance metrics using financial optimization libraries, and then clusters stocks based on their annualized returns and variances. It provides interactive visualizations and detailed cluster analyses, enabling users to gain insights into potential diversification strategies.

## Features

- **Historical Data Retrieval:**  
  Fetches stock data from Yahoo Finance for selected tickers using `yfinance`.
- **Financial Metrics & Optimization:**  
  Uses the PyPortfolioOpt library to calculate expected returns, sample covariance, and constructs an efficient frontier.
- **Clustering Analysis:**  
  Offers multiple clustering techniques for portfolio diversification:
  - **Classical Machine Learning:**
    - K-Means
    - Agglomerative Clustering
    - DBSCAN
    - Gaussian Mixture Model
  - **Quantum Machine Learning:**
    - QSVC (Quantum Support Vector Classification)
    - VQC (Variational Quantum Classifier)
- **Interactive Visualizations:**  
  Displays stock price charts, cluster plots, and detailed cluster breakdowns using Plotly.
- **Stock Details:**  
  Retrieves and displays stock-specific information (e.g., industry, CEO, country, revenue) using `yfinance`.

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/Jnan-py/Portfolio-Diversify.git
   cd Portfolio-Diversify
   ```

2. **Create a Virtual Environment (Recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows use: venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Run the Application:**

   ```bash
   streamlit run main.py
   ```

2. **Select Machine Learning Backend:**  
   Use the sidebar to choose between _Classical Machine Learning_ and _Quantum Machine Learning_ for clustering analysis.

3. **Input Parameters:**

   - Enter the number of tickers and choose the clustering strategy.
   - Select stock tickers from the provided list (fetched from the S&P 500 companies Wikipedia page).
   - View detailed stock information by selecting an individual ticker.

4. **Cluster Analysis:**
   - After entering the required inputs, click the button to generate clusters.
   - The app computes annualized returns and variances and then applies the chosen clustering algorithm.
   - Interactive plots and tables display cluster details, silhouette scores, and stock price graphs for further insights.

## Project Structure

```
portfolio-diversification-app/
│
├── main.py                 # Main Streamlit application
├── README.md              # Project documentation
├── requirements.txt       # Python dependencies
```

## Technologies Used

- **Streamlit:** Interactive web application framework.
- **Pandas & NumPy:** Data manipulation and numerical computations.
- **Yahoo Finance (yfinance):** Historical stock data retrieval.
- **PyPortfolioOpt:** Financial optimization, risk models, and efficient frontier construction.
- **Prophet:** Time series forecasting and visualization (if used for price trends).
- **Plotly:** Interactive plotting and data visualization.
- **Scikit-Learn:** Classical clustering algorithms (K-Means, Agglomerative, DBSCAN, Gaussian Mixture).
- **Qiskit & Qiskit Machine Learning:** Quantum machine learning algorithms (QSVC, VQC) for clustering.
