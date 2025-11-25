import gradio as gr
import pandas as pd
import plotly.express as px
from transformers import pipeline

classifier_pipeline = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def get_chart_type(prompt):
    """
    Analyzes the user's natural language prompt to determine chart type.
    
    Args:
        prompt (str): User's natural language description of the desired chart
        
    Returns:
        str: Chart type string (e.g., 'bar', 'line', 'histogram', 'scatter', 'donut')
    """
    candidate_labels = [
        "a comparison of different categories",
        "a trend or growth over a period of time",
        "the progression of a value as another continuos varible increases",
        "a relationship or correlation between two variables",
        "a distribution or frequency of one variable",
        "a composition or parts of a whole"
    ]

    hypothesis_template = "This text is asking for a chart about {}"
    prompt_lower = prompt.lower()
    result = classifier_pipeline(prompt, candidate_labels, hypothesis_template=hypothesis_template)
    best_label = result["labels"][0]
    print(f"scores - {result["scores"]}")
    print(f"User prompt - {prompt} -> Best label - {best_label}")
    label_to_chart_map = {
        "a comparison of different categories": "bar",
        "a trend or growth over a period of time": "line",
        "the progression of a value as another continuous varible increases":"line",
        "a relationship or correlation between two variables": "scatter",
        "a distribution or frequency of one variable": "histogram",
        "a composition or parts of a whole": "donut"
    }
    
    # # Determine chart type based on keywords
    # chart_type = None
    
    # # Comparison keywords -> bar chart
    # if any(keyword in prompt_lower for keyword in ['compare', 'comparison', 'vs', 'versus', 'by']):
    #     chart_type = 'bar'
    
    # # Trend keywords -> line chart
    # elif any(keyword in prompt_lower for keyword in ['trend', 'growth', 'over time', 'monthly', 'yearly', 'time series']):
    #     chart_type = 'line'
    
    # # Distribution keywords -> histogram
    # elif any(keyword in prompt_lower for keyword in ['distribution', 'spread', 'frequency', 'histogram']):
    #     chart_type = 'histogram'
    
    # # Relationship keywords -> scatter plot
    # elif any(keyword in prompt_lower for keyword in ['relationship', 'correlation', 'link between', 'scatter']):
    #     chart_type = 'scatter'
    
    # # Composition keywords -> donut/pie chart
    # elif any(keyword in prompt_lower for keyword in ['composition', 'part-to-whole', 'makeup', 'share of', 'percentage', 'pie']):
    #     chart_type = 'donut'
    
    # # Default to bar chart if no intent recognized
    # if chart_type is None:
    #     chart_type = 'bar'
    
    return label_to_chart_map.get(best_label, "bar")

def generate_visualization(file, prompt, column_names_str, trendline_choice):
    """
    Main function that generates visualizations based on uploaded CSV and user prompt.
    
    Args:
        file: Uploaded file object from Gradio
        prompt (str): User's natural language description of the desired chart
        column_names_str (str): Comma-separated column names
        
    Returns:
        plotly.graph_objects.Figure: Generated Plotly figure
    """
    # Check if file is uploaded
    if file is None:
        raise gr.Error("Please upload a CSV file first.")
    
    if not prompt or prompt.strip() == "":
        raise gr.Error("Please provide a description of the chart you want to see.")
    
    if not column_names_str or column_names_str.strip() == "":
        raise gr.Error("Please provide column names.")
    
    try:
        # Load data from CSV
        # Handle both file object and filepath string
        file_path = file.name if hasattr(file, 'name') else file
        df = pd.read_csv(file_path)
        
        # Get chart type from NLP analysis
        chart_type = get_chart_type(prompt)
        
        # Parse column names from the comma-separated string
        columns = [col.strip() for col in column_names_str.split(',')]
        
        # Limit columns based on chart type
        if chart_type == 'histogram':
            columns = columns[:1] if columns else []
        else:
            columns = columns[:2] if columns else []
        
        # If no columns provided, use first columns of dataframe as fallback
        if not columns:
            if chart_type == 'histogram':
                columns = [df.columns[0]] if len(df.columns) > 0 else []
            else:
                columns = list(df.columns[:2]) if len(df.columns) >= 2 else list(df.columns[:1])
        
        # Validate columns exist in dataframe
        available_columns = [col.lower() for col in df.columns]
        validated_columns = []
        
        for col in columns:
            # Try exact match first
            if col in df.columns:
                validated_columns.append(col)
            # Try case-insensitive match
            elif col.lower() in available_columns:
                idx = available_columns.index(col.lower())
                validated_columns.append(df.columns[idx])
            # Try partial match
            else:
                for df_col in df.columns:
                    if col.lower() in df_col.lower() or df_col.lower() in col.lower():
                        validated_columns.append(df_col)
                        break
        
        # If still no columns found, use first available columns
        if not validated_columns:
            if chart_type == 'histogram':
                validated_columns = [df.columns[0]] if len(df.columns) > 0 else []
            else:
                validated_columns = list(df.columns[:2]) if len(df.columns) >= 2 else [df.columns[0]]
        
        # Generate chart based on type
        fig = None
        
        if chart_type == 'bar':
            if len(validated_columns) >= 2:
                fig = px.bar(df, x=validated_columns[0], y=validated_columns[1],
                            color=validated_columns[0],
                            color_discrete_sequence=px.colors.qualitative.Set3,
                            title=f"Bar Chart: {validated_columns[1]} by {validated_columns[0]}")
            else:
                # If only one column, create a count bar chart
                fig = px.bar(df, x=validated_columns[0],
                            color=validated_columns[0],
                            color_discrete_sequence=px.colors.qualitative.Set3,
                            title=f"Bar Chart: Count of {validated_columns[0]}")
        
        elif chart_type == 'line':
            if len(validated_columns) >= 2:
                # Sort x and y values by x-axis column in ascending order
                df_sorted = df.sort_values(by=validated_columns[0], ascending=True)
                df_trend = df_sorted.groupby(validated_columns[0])[validated_columns[1]].mean().reset_index()
                # x_sorted = df_sorted[validated_columns[0]].values
                # y_sorted = df_sorted[validated_columns[1]].values
                # Create a new dataframe with sorted values
                # df_line = pd.DataFrame({validated_columns[0]: x_sorted, validated_columns[1]: y_sorted})
                fig = px.line(df_trend, x=validated_columns[0], y=validated_columns[1], markers=True,
                             title=f"Line Chart: {validated_columns[1]} over {validated_columns[0]}")
            else:
                # Sort by index for single column line chart
                df_sorted = df.sort_index(ascending=True)
                x_sorted = df_sorted.index.values
                y_sorted = df_sorted[validated_columns[0]].values
                # Create a new dataframe with sorted values
                df_line = pd.DataFrame({'index': x_sorted, validated_columns[0]: y_sorted})
                fig = px.line(df_line[12, :], x='index', y=validated_columns[0][12],
                             title=f"Line Chart: {validated_columns[0]} over time")
        
        elif chart_type == 'histogram':
            fig = px.histogram(df, x=validated_columns[0],
                              title=f"Histogram: Distribution of {validated_columns[0]}")
        
        elif chart_type == 'scatter':
            if len(validated_columns) >= 2:
                # Map user-friendly trendline choice to plotly trendline values
                trendline_map = {
                    "None": None,
                    "Linear (OLS)": "ols",
                    "LOWESS (Smoothed)": "lowess"
                }
                plotly_trendline_value = trendline_map.get(trendline_choice or "None", None)
                fig = px.scatter(
                    df,
                    x=validated_columns[0],
                    y=validated_columns[1],
                    color=validated_columns[0],
                    trendline=plotly_trendline_value,
                    title=f"Scatter Plot: {validated_columns[1]} vs {validated_columns[0]}"
                )
            else:
                raise ValueError("Scatter plot requires at least 2 columns")
        
        elif chart_type == 'donut':
            if len(validated_columns) >= 2:
                fig = px.pie(df, names=validated_columns[0], values=validated_columns[1],
                            hole=0.4, title=f"Donut Chart: {validated_columns[1]} by {validated_columns[0]}")
            else:
                # If only one column, create pie chart with counts
                value_counts = df[validated_columns[0]].value_counts()
                fig = px.pie(values=value_counts.values, names=value_counts.index,
                            hole=0.4, title=f"Donut Chart: Distribution of {validated_columns[0]}")
        
        if fig is None:
            raise ValueError("Could not generate chart. Please check your data and prompt.")
        
        # Calculate y-axis range from data to scale properly
        y_min = None
        y_max = None
        
        if chart_type in ['bar', 'line', 'scatter'] and len(validated_columns) >= 2:
            y_values = df[validated_columns[1]]
            y_min = float(y_values.min())
            y_max = float(y_values.max())
            # Add small padding (5% of range)
            padding = (y_max - y_min) * 0.05 if y_max != y_min else abs(y_max) * 0.05 if y_max != 0 else 1
            y_min = y_min - padding
            y_max = y_max + padding
        elif chart_type == 'line' and len(validated_columns) == 1:
            y_values = df[validated_columns[0]]
            y_min = float(y_values.min())
            y_max = float(y_values.max())
            padding = (y_max - y_min) * 0.05 if y_max != y_min else abs(y_max) * 0.05 if y_max != 0 else 1
            y_min = y_min - padding
            y_max = y_max + padding
        
        # Update layout for better appearance
        layout_updates = {
            'height': 500,
            'showlegend': True,
            'template': 'plotly_white'
        }
        
        # Scale y-axis to match data values
        if chart_type == 'bar':
            # Bar charts should start from 0
            if y_max is not None:
                layout_updates['yaxis'] = {'range': [0, y_max]}
            else:
                layout_updates['yaxis'] = {'range': [0, None]}
        elif chart_type in ['line', 'scatter'] and y_min is not None and y_max is not None:
            # For line and scatter plots, scale to data range
            layout_updates['yaxis'] = {'range': [y_min, y_max]}
        else:
            # For other chart types, use autorange
            layout_updates['yaxis'] = {'autorange': True}
        
        fig.update_layout(**layout_updates)
        
        return fig
    
    except pd.errors.EmptyDataError:
        raise gr.Error("The uploaded file is empty. Please upload a valid CSV file.")
    except pd.errors.ParserError:
        raise gr.Error("Error parsing CSV file. Please ensure the file is a valid CSV format.")
    except KeyError as e:
        raise gr.Error(f"Column not found in data: {e}. Available columns: {', '.join(df.columns)}")
    except ValueError as e:
        raise gr.Error(str(e))
    except Exception as e:
        raise gr.Error(f"An error occurred: {str(e)}")

# Create Gradio Interface
with gr.Blocks(title="AI-Powered Data Visualization Tool") as demo:
    gr.Markdown("# AI-Powered Data Visualization Tool")
    gr.Markdown("Upload a CSV file and describe the chart you want to see using natural language!")
    
    # Helper to dynamically show/hide trendline options based on prompt intent
    def update_trendline_visibility(prompt):
        chart_type = get_chart_type(prompt or "")
        if chart_type == 'scatter':
            return gr.update(visible=True)
        return gr.update(visible=False)
    
    with gr.Row():
        with gr.Column():
            file_input = gr.File(
                label="1. Upload your CSV",
                file_types=[".csv"],
                type="filepath"
            )
            prompt_input = gr.Textbox(
                label="2. Describe the Goal of Your Chart",
                placeholder="e.g., Show a comparison",
                lines=3
            )
            columns_input = gr.Textbox(
                label="3. Enter Column Names (comma-separated)",
                placeholder="e.g., Region, Sales",
                lines=2
            )
            trendline_input = gr.Radio(
                label="4. Add a Trendline (for Scatter Plots only)",
                choices=["None", "Linear (OLS)", "LOWESS (Smoothed)"],
                value="None",
                visible=False
            )
            generate_btn = gr.Button("Generate Chart", variant="primary", size="lg")
        
        with gr.Column():
            plot_output = gr.Plot(label="Generated Visualization")
    
    # Add some examples
    gr.Markdown("### Example Prompts:")
    gr.Markdown("""
    - "Compare sales by region" → Bar chart
    - "Show me the trend of revenue over time" → Line chart
    - "Display the distribution of ages" → Histogram
    - "Show the relationship between price and quantity" → Scatter plot
    - "What's the composition of sales by product?" → Donut chart
    """)
    
    # Show/hide trendline options based on prompt updates
    prompt_input.submit(
        fn=update_trendline_visibility,
        inputs=prompt_input,
        outputs=trendline_input,
    )
    
    # Connect the button to the function
    generate_btn.click(
        fn=generate_visualization,
        inputs=[file_input, prompt_input, columns_input, trendline_input],
        outputs=plot_output
    )

# Launch the application
if __name__ == "__main__":
    demo.launch(share=True, server_name="0.0.0.0", server_port=7860)

