# %%
import pandas as pd

data = pd.read_csv('../datasets/mod_05_topic_09_employee_data.csv')

# %%
data.head()
data.info()

# %%
#work experience in the company

data['JoiningYear'] = data['JoiningYear'].max() - data['JoiningYear']

# %%
#convert num column to categorial

data['PaymentTier'] = data['PaymentTier'].astype(str)

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = (
    train_test_split(
        data.drop('LeaveOrNot', axis=1),
        data['LeaveOrNot'],
        test_size=0.33,
        random_state=42))

# %%
# coding of categorial columns
from category_encoders import TargetEncoder

encoder = TargetEncoder()

X_train = encoder.fit_transform(X_train, y_train)
X_test = encoder.transform(X_test)

# %%
# normalize the data
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().set_output(transform='pandas')

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# %%
#check class' balance

y_train.value_counts(normalize=True)

# %%
# balanced the data
from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=42)

X_res, y_res = sm.fit_resample(X_train, y_train)

# %%
#helper to save the time, f1_score of ensembles
from sklearn.metrics import f1_score
from timeit import time

f1_scores = {}

def measure_f1_time_decorator(fn):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        predictions = fn(*args, **kwargs)
        end_time = time.time()
        f1 = f1_score(args[-1], predictions)
        model_name = args[0].__class__.__name__
        execution_time = end_time - start_time
        f1_scores[model_name] = [f1, execution_time]
        
        print(f'{model_name} F1 Metric: {f1:.4f}')
        print(f'{model_name} Inference: {execution_time:.4f} s')
        
        return predictions
    
    return wrapper
        

@measure_f1_time_decorator
def predict_with_measure(model, Xt, yt):
    return model.predict(Xt)

# %%
#base algo
from sklearn.linear_model import LogisticRegression

mod_log_reg = LogisticRegression().fit(X_res, y_res)

prd_log_reg = predict_with_measure(mod_log_reg, X_test, y_test)

# %%
# deep trees
from sklearn.ensemble import RandomForestClassifier

ran_for_clf = RandomForestClassifier(random_state=42).fit(X_res, y_res)

prd_ran_for = predict_with_measure(ran_for_clf, X_test, y_test)

# %%
# bagging classifier
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier

bagging_clf = BaggingClassifier(KNeighborsClassifier(),
                                max_features=0.75,
                                max_samples=0.75,
                                random_state=42).fit(X_res, y_res)

prd_bagging_clf = predict_with_measure(bagging_clf, X_test, y_test)

# %%
# ada boosting clf
from sklearn.ensemble import AdaBoostClassifier

ada_clf = AdaBoostClassifier(algorithm='SAMME', random_state=42).fit(X_res, y_res)

prd_ada_clf = predict_with_measure(ada_clf, X_test, y_test)

# %%
# gradient boosting
from sklearn.ensemble import GradientBoostingClassifier

gradient_clf = GradientBoostingClassifier(
    learning_rate=0.3,
    max_features='sqrt',
    subsample=0.75,
    random_state=42,).fit(X_res, y_res)

prd_gradient  = predict_with_measure(gradient_clf, X_test, y_test)

# %%
#voting clf
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import GaussianNB

clf1 = GaussianNB()
clf2 = KNeighborsClassifier()
clf3 = LogisticRegression()

estimators = [('lnr', clf3),
              ('knn', clf2),
              ('gnb', clf1)]

voting_clf = VotingClassifier(estimators=estimators, 
                              voting='soft').fit(X_res, y_res)

prd_voting = predict_with_measure(voting_clf, X_test, y_test)

# %%
# stacking clf, metamoodel
from sklearn.ensemble import StackingClassifier

final_estimator = GradientBoostingClassifier(
    subsample=0.75,
    max_features='sqrt',
    random_state=42)

stacking_clf = StackingClassifier(
    estimators=estimators, 
    final_estimator=final_estimator).fit(X_res, y_res)

prd_stack = predict_with_measure(stacking_clf, X_test, y_test)

# %%
# compare ensembles

scores = pd.DataFrame().from_dict(
    f1_scores,
    orient='index',
    columns=['f1', 'time'])

scores.sort_values('f1', ascending=False)



