# Smart MCQ Solver Challenge - Deep Learning and Gen AI
Build AI systems that solve & predict top 3 correct answers for challenging multiple choice questions

Name: Jitendra Rathore
ID: 23f3003059

Deployment Link: [mcq-solver](https://mcq-deployment.streamlit.app/)

## Project structure
```
├── Deployment
│   ├── app.py
│   ├── final-model
│   ├── label_encoder.pkl
│   ├── mlp_model.pth
│   ├── requirements.txt
│   ├── tfidf_vectorizer.pkl
│   ├── venv
│   └── xgb_model.pkl
├── DL-23f3003059-notebook-t22026.ipynb
└── README.md
```

We have 6 branches, where 5 branches are for milestones and 1 main branch where we have all the code for this competition. We have given two csv files train and test. train.csv contains training data along with answer labels
and test.csv contains test data without answers, we have to predict answer for this file.

train.csv: id, prompt, A, B, C, D, E, answer
test.csv: id, prompt, A, B, C, D, E

## Tech used
- Python
- Numpy
- Pandas
- Matplotlib
- Pytorch


## Installation
```
git clone https://github.com/23f3003059/dlgenai-t22026.git
cd dlgenai-t22026/Deployment


# install dependecies
python -m venv
source .venv/bin/activate
pip install -r requirements.txt


# run project
streamlit run app.py
```







