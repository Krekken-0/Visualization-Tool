# AI-Powered Data Visualization Tool

An intelligent web application that generates data visualizations from CSV files using natural language processing. Simply describe the chart you want to see in plain English, and the tool automatically determines the best chart type and creates an interactive visualization.

## Features

- 🤖 **AI-Powered Chart Selection**: Uses zero-shot classification (BART-large-MNLI) to intelligently determine chart types from natural language prompts
- 📊 **Multiple Chart Types**: Supports bar charts, line charts, histograms, scatter plots, and donut charts
- 📁 **CSV File Upload**: Easy drag-and-drop CSV file upload
- 🔍 **Smart Column Matching**: Automatically matches column names with case-insensitive and partial matching
- 📈 **Interactive Visualizations**: Powered by Plotly for interactive, publication-quality charts
- 🎯 **Trendline Support**: Optional trendlines (Linear OLS or LOWESS) for scatter plots
- 🌐 **Web Interface**: User-friendly Gradio-based web interface

## Technology Stack

- **Gradio**: Web interface framework
- **Pandas**: Data processing and manipulation
- **Plotly**: Interactive visualization library
- **Transformers (Hugging Face)**: Natural language processing using BART-large-MNLI model
- **Statsmodels**: Statistical modeling for trendline calculations

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Krekken-0/Visualization-Tool.git
cd Visualization-Tool
```

2. Create a virtual environment (recommended):
```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
pip install transformers torch  # Additional dependencies for NLP model
```

## Usage

1. Start the application:
```bash
gradio visualization_tool.py
```

2. Open your web browser and navigate to the URL shown in the terminal (typically `http://localhost:7860` or a Gradio share link)

3. Use the interface:
   - **Upload CSV**: Click or drag-and-drop your CSV file
   - **Describe Chart**: Enter a natural language description of the chart you want (e.g., "Show me a comparison of sales by region")
   - **Enter Columns**: Provide comma-separated column names from your CSV (e.g., "Region, Sales")
   - **Generate**: Click the "Generate Chart" button

4. The tool will automatically:
   - Analyze your prompt to determine the chart type
   - Match and validate column names
   - Generate an interactive visualization

## Supported Chart Types

The tool automatically selects chart types based on your prompt:

| Chart Type | Use Case | Example Prompt |
|------------|----------|----------------|
| **Bar Chart** | Comparisons between categories | "Compare sales by region" |
| **Line Chart** | Trends over time | "Show me the trend of revenue over time" |
| **Histogram** | Distribution of a single variable | "Display the distribution of ages" |
| **Scatter Plot** | Relationships between two variables | "Show the relationship between price and quantity" |
| **Donut Chart** | Composition or parts of a whole | "What's the composition of sales by product?" |

## Example Prompts

- "Compare sales by region" → Bar chart
- "Show me the trend of revenue over time" → Line chart
- "Display the distribution of ages" → Histogram
- "Show the relationship between price and quantity" → Scatter plot
- "What's the composition of sales by product?" → Donut chart

## How It Works

1. **Natural Language Understanding**: The tool uses Facebook's BART-large-MNLI model to classify user prompts into chart intent categories using zero-shot classification.

2. **Chart Type Mapping**: The classified intent is mapped to an appropriate chart type:
   - Comparison → Bar chart
   - Trend/Growth → Line chart
   - Distribution → Histogram
   - Relationship/Correlation → Scatter plot
   - Composition → Donut chart

3. **Data Processing**: The CSV file is loaded and columns are validated with intelligent matching (exact, case-insensitive, and partial matching).

4. **Visualization Generation**: Plotly generates an interactive chart with appropriate styling, axis scaling, and formatting.

## Configuration

The application runs on:
- **Host**: `0.0.0.0` (accessible from all network interfaces)
- **Port**: `7860`
- **Share**: Enabled (creates a public Gradio link)

To modify these settings, edit the `demo.launch()` call in `visualization_tool.py`:

```python
demo.launch(share=False, server_name="127.0.0.1", server_port=7860)
```

## Requirements

See `requirements.txt` for the complete list of dependencies:

- gradio>=4.0.0
- pandas>=2.0.0
- plotly>=5.0.0
- statsmodels>=0.13.0
- transformers (for NLP model)
- torch (required by transformers)

## Notes

- The first run will download the BART-large-MNLI model (~1.6GB), which may take some time
- CSV files should have a header row with column names
- Column names are matched intelligently (case-insensitive, partial matching supported)
- For scatter plots, you can optionally add trendlines (Linear OLS or LOWESS smoothed)


