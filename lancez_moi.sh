#!/bin/bash


#python -m venv venvtest

source venv/bin/activate
#source venvtest/bin/activate
#pip install -r requirements.txt

cd results

jupyter nbconvert --execute --to html metrics.ipynb && open metrics.html #un notebook avec les resultats a presenter

jupyter nbconvert --execute --to html explainability.ipynb && open explainability.html 

cd ..

cd src

cd models

streamlit run CNN_brain_tumor_saved.py # a mettre en dernier. pour faire joue joue avce streamlit 

#./lancez_moi.sh