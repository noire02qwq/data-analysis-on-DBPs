
import pandas as pd

def create_latex_table(df, target_name, caption, label):
    df_target = df[df['target'].str.contains(target_name)].copy()
    # Pre-process model and mode columns
    df_target['Model'] = df_target['model'].str.upper() + '-' + df_target['mode'].str.capitalize()
    
    # Drop unnecessary columns and sort
    df_target = df_target.drop(columns=['target', 'mode', 'trials-id', 'model'])
    df_target = df_target.sort_values(by='mse')
    
    # Reorder columns
    df_target = df_target[['Model', 'mse', 'rmse', 'mae']]
    
    metrics = ['mse', 'rmse', 'mae']
    min_vals = {metric: df_target[metric].min() for metric in metrics}
    
    for metric in metrics:
        # Format values, highlighting the minimum
        df_target[metric] = df_target[metric].apply(
            lambda x: f'\textbf{{\textcolor{{red}}{{{x:.6f}}}}}' if x == min_vals[metric] else f'{x:.6f}'
        )

    # Build the LaTeX table string
    latex_table = f'\begin{{table}}[h!]
'
    latex_table += f'\centering
'
    latex_table += f'\caption{{{caption}}}
'
    latex_table += f'\label{{{label}}}
'
    latex_table += df_target.to_latex(index=False, escape=False, column_format='lrrr', header=['Model', 'MSE', 'RMSE', 'MAE'])
    latex_table += f'\end{{table}}
'
    
    return latex_table

# Load the data
df = pd.read_csv('Archived/midterm/result.csv')

# Generate tables
trc_table = create_latex_table(df, 'trc', 'Performance Metrics for TRC Prediction at PPL2', 'tab:results_trc')
ph_table = create_latex_table(df, 'ph', 'Performance Metrics for pH Prediction at PPL2', 'tab:results_ph')
toc_table = create_latex_table(df, 'toc', 'Performance Metrics for TOC Prediction at PPL2', 'tab:results_toc')

# Print tables to be captured by the shell
print('%%%%%%%%%%%%%%%%%%%% TRC Table %%%%%%%%%%%%%%%%%%%%')
print(trc_table)
print('%%%%%%%%%%%%%%%%%%%% pH Table %%%%%%%%%%%%%%%%%%%%%')
print(ph_table)
print('%%%%%%%%%%%%%%%%%%%% TOC Table %%%%%%%%%%%%%%%%%%%%')
print(toc_table)
