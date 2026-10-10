# Resume Screening: Job-Category Classification

Classifies a resume's raw text into one of 25 job categories (for example Data Science, HR, Java Developer
or Database), as a first automated step in screening.

## Pipeline

1. **Data:** `resume_dataset.csv` has 169 resumes in 25 categories, with columns `Category` and `Resume`.
2. **Cleaning:** regular expressions remove URLs, RT/cc markers, hashtags, mentions, punctuation, non-ASCII
   characters and extra whitespace.
3. **Features:** TF-IDF with sublinear term frequency, English stop words removed, and the top 1,500
   features kept.
4. **Model:** a one-vs-rest k-nearest-neighbours classifier, evaluated with accuracy and a per-category
   classification report.

## Results

| Split | Accuracy |
|---|---|
| Train | 0.88 |
| Test | **0.79** |

With only 169 resumes spread over 25 classes, many categories have fewer than 10 examples. The results are
therefore a baseline. More data, or embeddings from a pretrained language model, are the obvious next step.

## Files

| File | Contents |
|---|---|
| `Resume_Screening.ipynb` | Exploration, cleaning, features, model and evaluation |
| `resume_dataset.csv` | Labelled resumes |

## Run

```bash
pip install pandas numpy scikit-learn matplotlib seaborn nltk
jupyter notebook Resume_Screening.ipynb
```
