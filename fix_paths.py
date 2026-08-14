import json

def update_notebook_paths(notebook_path):
    with open(notebook_path, 'r') as f:
        nb = json.load(f)
    
    # 1. Create the new config cell
    config_code = [
        "import os\n",
        "\n",
        "# --- PATH CONFIGURATION ---\n",
        "# Set this to True if you are running in Google Colab\n",
        "RUNNING_IN_COLAB = True\n",
        "\n",
        "if RUNNING_IN_COLAB:\n",
        "    # By default, Colab uploads go to the current directory (/content/)\n",
        "    TRAIN_PATH = 'train_dataset.csv'\n",
        "    TEST_PATH = 'test_dataset.csv'\n",
        "    PREDS_DIR = '.'\n",
        "else:\n",
        "    # Default local paths\n",
        "    TRAIN_PATH = '../datasets/train_dataset.csv'\n",
        "    TEST_PATH = '../datasets/test_dataset.csv'\n",
        "    PREDS_DIR = '../predictions'\n",
        "\n",
        "os.makedirs(PREDS_DIR, exist_ok=True)\n"
    ]
    
    config_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": config_code
    }
    
    # Insert config cell after the first code cell (which contains the imports)
    insert_idx = 0
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code":
            insert_idx = i + 1
            break
            
    nb["cells"].insert(insert_idx, config_cell)
    
    # 2. Update usages in all other code cells
    for cell in nb["cells"]:
        if cell["cell_type"] == "code" and cell is not config_cell:
            new_source = []
            for line in cell["source"]:
                # Replace Train path
                if "'../datasets/train_dataset.csv'" in line:
                    line = line.replace("'../datasets/train_dataset.csv'", "TRAIN_PATH")
                if '"../datasets/train_dataset.csv"' in line:
                    line = line.replace('"../datasets/train_dataset.csv"', "TRAIN_PATH")
                    
                # Replace Test path
                if "'../datasets/test_dataset.csv'" in line:
                    line = line.replace("'../datasets/test_dataset.csv'", "TEST_PATH")
                if '"../datasets/test_dataset.csv"' in line:
                    line = line.replace('"../datasets/test_dataset.csv"', "TEST_PATH")
                
                # Replace Prediction Output paths
                if "'../predictions/xgboost_pred2.csv'" in line:
                    line = line.replace("'../predictions/xgboost_pred2.csv'", "f'{PREDS_DIR}/xgboost_pred2.csv'")
                if "'../predictions/GPR_predictions.csv'" in line:
                    line = line.replace("'../predictions/GPR_predictions.csv'", "f'{PREDS_DIR}/GPR_predictions.csv'")
                
                # Remove old makedirs since it's in the config block now
                if "os.makedirs('../predictions'" in line:
                    continue 
                
                new_source.append(line)
            cell["source"] = new_source
            
    with open(notebook_path, 'w') as f:
        json.dump(nb, f, indent=1)

update_notebook_paths('/Users/shard/projects/ML_hackathon_KGP/XG-Boost/xgboost_nbk2.ipynb')
update_notebook_paths('/Users/shard/projects/ML_hackathon_KGP/GPR/ML_Hackathon_GPR_Final.ipynb')
print("Successfully made paths flexible for Colab in both notebooks!")
