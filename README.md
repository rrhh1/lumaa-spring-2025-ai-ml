### Ray Hu
# AI/Machine Learning Intern Challenge: Simple Content-Based Recommendation Submission

## Dataset
The dataset is from [Kaggle](https://www.kaggle.com/datasets/yehorkorzh/imdb-top-250-movies): It has 250 rows and 21 column features

## Setup
Developed using Python 3.11 <br>
run `pip install -r requirements.txt` in the terminal for the required packages

## Running
Run `python main.py` in the terminal. <br>
A prompt message will appear in the terminal. Enter your query and hit enter. Program will take query and return the recommendation results along with their cosine similarity scores. Prompt will loop until user enters 'exit'.

## Results
### Sample Results
Given query `"Spiderman movies"`
```
Results
Movie: Spider-Man: Homecoming --- Similarity: 0.25246131635410995
Movie: Jaws --- Similarity: 0.09171447462923983
Movie: One Flew Over the Cuckoo's Nest --- Similarity: 0.05380218236136394
Movie: Batman Begins --- Similarity: 0.05318890456620466
```
<br>

Given query `"Movies where explorers leave Earth"`
```
Results
Movie: Interstellar --- Similarity: 0.10067269827362035
Movie: Heat --- Similarity: 0.08207040109186725
Movie: The Martian --- Similarity: 0.06437958030959332
Movie: Blade Runner --- Similarity: 0.059875181722008085
Movie: The Lord of the Rings: The Fellowship of the Ring --- Similarity: 0.05933759453732401
```
### Video Demonstrations
Video Demonstration of `main.py` found in `demo.md`. <br>

### Deeper Analysis
Deeper Analysis and further experiments found in `just_for_fun.ipynb`

## Salary Expectations
Depends on current funding, and size of the startup.  
Within the range of [$1600, $2400] per month (based on the $20 - $30 per hour listing)



