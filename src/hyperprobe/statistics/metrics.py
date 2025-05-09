from os import makedirs, path
import json

# LOCAL IMPORTS
from hyperprobe.statistics import utils
        
if __name__ == '__main__':
    
    root_path = path.join('outputs', 'probing')
    
    # Load the results
    file_name = 'llama4_13apr_verbose.json'
    with open(path.join(root_path, file_name), 'r') as file:
        results = json.load(file)
    
    # Import the results
    df = utils.load_results(results) 

    # Create the output folder
    versionName = path.splitext(file_name)[0]
    root_output_folder = path.join(root_path, 'stats', versionName)
    
     # Create the output folder
    obs_output_folder = path.join(root_output_folder, 'observations')
    makedirs(obs_output_folder, exist_ok = True)
    
    # Compute the blank scenario
    item_per_area = df.groupby('area')['doc'].size().to_dict()
    print('ITEMS PER AREA:', item_per_area) 
    
    blank_scenario = df[(df['next_token_precision@5'] == 0) & (df['precision@1'] == 0)]
    blank_scenario = blank_scenario.groupby('area', as_index=False)['doc'].size()
    blank_scenario['size'] = blank_scenario.apply(
        lambda row: round((row['size'] / item_per_area[row['area']]) * 100, 1), axis = 1)
    blank_scenario = blank_scenario.sort_values(by = 'size', ascending = False)
    
    print('BLANK SCENARIO (%):\n', blank_scenario)
    
    # Visualize the observations
    utils.view_observations(df, name = 'ALL', output_folder = obs_output_folder)
    for col in ['area', 'unbunding_type']:
        for label in df[col].unique():
            to_view = df[df[col] == label]
                 
            if len(to_view) == 0:
                continue
            
            # Create the output folder
            col_output_folder = path.join(obs_output_folder, col)
            makedirs(col_output_folder, exist_ok = True)
            
            # Sort the columns
            sorted_cols = sorted(to_view.columns.to_list(), key = lambda x: df.columns.get_loc(x) if x in df.columns else 1000)
            to_view = to_view[sorted_cols]

            # Save the observations
            utils.view_observations(to_view, name = label, output_folder = col_output_folder)
    
    # Isolate the wrong predictions
    cond = df['next_token_precision@5'] == 0 #& (df['precision@5'] == 1)
    wrong_df = df[cond].drop(columns = ['next_token_precision@1'])
    utils.view_observations(wrong_df, name = 'WRONG_PREDICTIONS', output_folder = obs_output_folder)
    
    print('WRONG DATASET:', round((len(wrong_df) / len(df) ) * 100, 1), '%')
    
    # Compute stats based on the domain and area
    utils.descriptive_stats(df, root_output_folder)
   
    # Compute correlations
    utils.compute_correlations(df.copy(), root_output_folder)

    # Create the graphs
    utils.create_graphs(df, root_output_folder)