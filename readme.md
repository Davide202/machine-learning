

Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser -Force

.\setup_env.ps1

cd machine-learning

pip install ipykernel jupyter pandas numpy matplotlib scikit-learn ucimlrepo seaborn 

 python -m pip install graphviz

python -m pip install pydot
