# Customer Segmentation Using Soft Computing (Fuzzy C-Means, GA, & ANN)

An end-to-end data science framework designed to segment customers based on their historical buying habits. The project processes transaction records into **RFM (Recency, Frequency, Monetary)** profiles, clusters customers using soft boundaries (**Fuzzy C-Means**), optimizes the features using a **Genetic Algorithm (GA)**, and trains an **Artificial Neural Network (ANN)** to predict customer classes.

---

## Tech Stack & Libraries
* **Language:** Python 3
* **Environment:** Google Colab / Jupyter Notebook
* **Data Processing:** `pandas`, `numpy`, `scikit-learn`
* **Soft Computing Clustering:** `scikit-fuzzy` (`skfuzzy`)
* **Heuristic Optimization:** Custom Genetic Algorithm using `RandomForestClassifier`
* **Deep Learning Layer:** `tensorflow.keras` (Sequential, Dense)
* **Data Visualization:** `matplotlib`, `seaborn`

---

## Dataset & Feature Engineering
The project uses the `online_retail_II.csv` dataset to build and engineer the following customer behavioral features:

| Feature Column | Description |
| :--- | :--- |
| `Recency` | Days since the customer's last purchase. |
| `Frequency` | Total number of unique purchase invoices. |
| `Monetary` | Total spending volume across the entire history. |
| `Log_Recency` / `_Frequency` / `_Monetary` | Log-transformed metrics to normalize skewed spending profiles. |
| `R_times_F` | Interaction term mapping $Recency \times Frequency$. |
| `M_over_F` | Average spending value per transaction ($\frac{Monetary}{Frequency + 1}$). |
| `R_plus_M` | Combined intensity factor ($Recency + Monetary$). |
| **`Cluster` (Target)** | Soft clustering category assigned by Fuzzy C-Means (`0`, `1`, or `2`). |

---

## How the Pipeline Works

### 1. RFM Engineering & Clean Up
* Drops blank customer entries and filters out negative quantities or prices.
* Groups transactions by customer to calculate core RFM values.

### 2. Fuzzy C-Means (FCM) Clustering
* Uses fuzzy boundaries to handle overlapping customer behaviors instead of rigid hard splits.
* Groups shoppers into **3 target segments** and assigns probabilistic membership weights to each.

### 3. Feature Selection Wrapper (Genetic Algorithm)
* Runs an evolutionary loop over **15 generations** with a population size of 10 and a 10% mutation rate.
* Automatically selects the best collection of feature combinations that maximizes classification performance.

### 4. Classification Deep Learning System (ANN)
* Feeds the GA-optimized features into a fully connected Sequential neural network:
  * **Hidden Layer 1:** 16 Neurons (ReLU)
  * **Hidden Layer 2:** 8 Neurons (ReLU)
  * **Output Layer:** 3 Neurons (Softmax) matching the target segments
* Trains over **50 epochs** using the `adam` optimizer to accurately classify user segments.

---

## Getting Started & Execution

### 1. Clone the Repository
```bash
git clone [https://github.com/YOUR_GITHUB_USERNAME/Customer-Segmentation-Soft-Computing.git](https://github.com/YOUR_GITHUB_USERNAME/Customer-Segmentation-Soft-Computing.git)
cd Customer-Segmentation-Soft-Computing
```
### 2. Install Required Dependencies
Bash
pip install pandas numpy scikit-learn scikit-fuzzy tensorflow matplotlib seaborn

### 3. Launch the Project
Open the notebook in your preferred environment (Jupyter or Google Colab) and run the pipeline cells sequentially:

```Bash
jupyter notebook "TCI Project Source_Code (Group60).ipynb"
```
📂 Project Structure
Plaintext
├── TCI Project Source_Code (Group60).ipynb   # Comprehensive project notebook containing all code blocks
├── README.md                                  # Clear project summary and setup manual
├── rfm_fuzzy_clustered.csv                     # Intermediary ledger tracking customer soft cluster clusters
└── final_dataset_for_ann.csv                  # Optimized dataset keeping features selected by the
