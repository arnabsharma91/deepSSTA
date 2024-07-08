# https://github.com/arnabsharma91/deepSSTA.git
Repository implementing the approach of team UPB-DICE

## Installation
- Clone this repository: `git clone https://github.com/arnabsharma91/deepSSTA.git`
- Create a conda environment named: `ecmldeep`: `conda create -n ecmldeep python=3.10.14`
- Activate the environment: `conda activate ecmlpdeep`
- Install all dependencies: `pip install -r requirements.txt`
- If working on Jupyter Lab or Jupyter Notebook, also install the kernel: `conda install ipykernel && python -m ipykernel install --user --name ecmldeep --display-name "ecmldeep"`

## Training our models:
- Run the notebooks `Train_LightGBM_CatBoost.ipynb`, `Train_LightGBM_CatBoost_more.ipynb`, `NN.ipynb`, and `NN_more.ipynb`

## Predicting for the Development Phase
- Run the notebook `Predict_Development_Phase.ipynb`

## Predicting for the Testing Phase
- Run the notebook `Predict_Test_Phase.ipynb`

## Predicting for September 2024
- Run the notebook `Predict_September2024.ipynb`

*Remark.* Our scores for both the development phase (0.09913) and testing phase (0.0872) are clearly visible in the notebooks `Predict_Development_Phase.ipynb` and `Predict_Test_Phase.ipynb`.
