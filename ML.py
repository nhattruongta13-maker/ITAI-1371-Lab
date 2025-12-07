import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score


df = pd.read_csv("heart.csv")



df['age_group'] = pd.cut(df['age'], bins=[0, 35, 45, 55, 65, 100],
                         labels=['<35', '35-44', '45-54', '55-64', '65+'])
if 'chol' in df.columns:
    df['chol_per_age'] = df['chol'] / df['age']
if 'trestbps' in df.columns and 'chol' in df.columns:
    df['bp_chol_ratio'] = df['trestbps'] / df['chol']

for col in df.columns:
    if df[col].dtype == 'object' or str(df[col].dtype) == 'category':
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        df[col] = df[col].fillna(df[col].mean())

cat_cols = df.select_dtypes(include=['object', 'category']).columns
for col in cat_cols:
    if df[col].nunique() <= 2:
        df[col] = LabelEncoder().fit_transform(df[col])
    else:
        df = pd.get_dummies(df, columns=[col], drop_first=True)

num_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[num_cols] = StandardScaler().fit_transform(df[num_cols])
df[num_cols] = MinMaxScaler().fit_transform(df[num_cols])


x_train, x_test, y_train, y_test = train_test_split(df.iloc[:, [x for x in range(0,16)]],
                                                    df.iloc[:, [x for x in range(16,20)]],
                                                    train_size=0.7, test_size=0.15, random_state=42,
                                                    shuffle=False)

x_validate, y_validate = (df.iloc[[x for x in range(871, len(df))], [x for x in range(0,16)]],
                          df.iloc[[x for x in range(871, len(df))], [x for x in range(16,20)]])


#train for age groups
y_train_for_35_44 = y_train.iloc[:, [0]]
y_train_for_35_44 = np.array(y_train_for_35_44).ravel()

y_train_for_45_54 = y_train.iloc[:, [1]]
y_train_for_45_54 = np.array(y_train_for_45_54).ravel()

y_train_for_55_64 = y_train.iloc[:, [2]]
y_train_for_55_64 = np.array(y_train_for_55_64).ravel()

y_train_for_65_more = y_train.iloc[:, [3]]
y_train_for_65_more = np.array(y_train_for_65_more).ravel()

#test for age groups
y_test_for_35_44 = y_test.iloc[:, [0]]
y_test_for_35_44 = np.array(y_test_for_35_44).ravel()

y_test_for_45_54 = y_test.iloc[:, [1]]
y_test_for_45_54 = np.array(y_test_for_45_54).ravel()

y_test_for_55_64 = y_test.iloc[:, [2]]
y_test_for_55_64 = np.array(y_test_for_55_64).ravel()

y_test_for_65_more = y_test.iloc[:, [3]]
y_test_for_65_more = np.array(y_test_for_65_more).ravel()

#validate for age groups
y_validate_for_35_44 = y_validate.iloc[:, [0]]
y_validate_for_45_54 = y_validate.iloc[:, [1]]
y_validate_for_55_64 = y_validate.iloc[:, [2]]
y_validate_for_65_more = y_validate.iloc[:, [3]]

#Initiate models
logistic_regression = LogisticRegression()
decision_tree = DecisionTreeClassifier(max_depth=200, max_leaf_nodes=50)
random_forest = RandomForestClassifier(max_depth=50, max_leaf_nodes=25)
bgm = AdaBoostClassifier(estimator=decision_tree, n_estimators=100, learning_rate=0.1)
k_neighbors = KNeighborsClassifier(n_neighbors=10, n_jobs=2)
support_vector = SVC()

#logistc regression train and validate
logistic_regression.fit(x_train,y_train_for_35_44)
logistic_regression_validate_35_44 = logistic_regression.predict(x_validate)
accuracy_logistic_1 = round(accuracy_score(y_validate_for_35_44, logistic_regression_validate_35_44),2)
precision_logistic_1 = round(precision_score(y_validate_for_35_44, logistic_regression_validate_35_44, zero_division=0),2)
recall_logistic_1 = round(recall_score(y_validate_for_35_44, logistic_regression_validate_35_44),2)
f1_logistic_1 = round(f1_score(y_validate_for_35_44, logistic_regression_validate_35_44),2)
roc_logistic_1 = round(roc_auc_score(y_validate_for_35_44, logistic_regression_validate_35_44),2)

logistic_regression.fit(x_train,y_train_for_45_54)
logistic_regression_validate_45_54 = logistic_regression.predict(x_validate)
accuracy_logistic_2 = round(accuracy_score(y_validate_for_45_54, logistic_regression_validate_45_54),2)
precision_logistic_2 = round(precision_score(y_validate_for_45_54, logistic_regression_validate_45_54, zero_division=0),2)
recall_logistic_2 = round(recall_score(y_validate_for_45_54, logistic_regression_validate_45_54),2)
f1_logistic_2 = round(f1_score(y_validate_for_45_54, logistic_regression_validate_45_54),2)
roc_logistic_2 = round(roc_auc_score(y_validate_for_45_54, logistic_regression_validate_45_54),2)

logistic_regression.fit(x_train,y_train_for_55_64)
logistic_regression_validate_55_64 = logistic_regression.predict(x_validate)
accuracy_logistic_3 = round(accuracy_score(y_validate_for_55_64, logistic_regression_validate_55_64),2)
precision_logistic_3 = round(precision_score(y_validate_for_55_64, logistic_regression_validate_55_64, zero_division=0),2)
recall_logistic_3 = round(recall_score(y_validate_for_55_64, logistic_regression_validate_55_64),2)
f1_logistic_3 = round(f1_score(y_validate_for_55_64, logistic_regression_validate_55_64),2)
roc_logistic_3 = round(roc_auc_score(y_validate_for_55_64, logistic_regression_validate_55_64),2)

logistic_regression.fit(x_train,y_train_for_65_more)
logistic_regression_validate_65_more = logistic_regression.predict(x_validate)
accuracy_logistic_4 = round(accuracy_score(y_validate_for_65_more, logistic_regression_validate_65_more),2)
precision_logistic_4 = round(precision_score(y_validate_for_65_more, logistic_regression_validate_65_more, zero_division=0),2)
recall_logistic_4 = round(recall_score(y_validate_for_65_more, logistic_regression_validate_65_more),2)
f1_logistic_4 = round(f1_score(y_validate_for_65_more, logistic_regression_validate_65_more),2)
roc_logistic_4 = round(roc_auc_score(y_validate_for_65_more, logistic_regression_validate_65_more),2)

#decision tree train and validate
decision_tree.fit(x_train,y_train_for_35_44)
decision_tree_validate_35_44 = decision_tree.predict(x_validate)
accuracy_decision_tree_1 = round(accuracy_score(y_validate_for_35_44, decision_tree_validate_35_44),2)
precision_decision_tree_1 = round(precision_score(y_validate_for_35_44, decision_tree_validate_35_44, zero_division=0),2)
recall_decision_tree_1 = round(recall_score(y_validate_for_35_44, decision_tree_validate_35_44),2)
f1_decision_tree_1 = round(f1_score(y_validate_for_35_44, decision_tree_validate_35_44),2)
roc_decision_tree_1 = round(roc_auc_score(y_validate_for_35_44, decision_tree_validate_35_44),2)

decision_tree.fit(x_train,y_train_for_45_54)
decision_tree_validate_45_54 = decision_tree.predict(x_validate)
accuracy_decision_tree_2 = round(accuracy_score(y_validate_for_45_54, decision_tree_validate_45_54),2)
precision_decision_tree_2 = round(precision_score(y_validate_for_45_54, decision_tree_validate_45_54, zero_division=0),2)
recall_decision_tree_2 = round(recall_score(y_validate_for_45_54, decision_tree_validate_45_54),2)
f1_decision_tree_2 = round(f1_score(y_validate_for_45_54, decision_tree_validate_45_54),2)
roc_decision_tree_2 = round(roc_auc_score(y_validate_for_45_54, decision_tree_validate_45_54),2)

decision_tree.fit(x_train,y_train_for_55_64)
decision_tree_validate_55_64 = decision_tree.predict(x_validate)
accuracy_decision_tree_3 = round(accuracy_score(y_validate_for_55_64, decision_tree_validate_55_64),2)
precision_decision_tree_3 = round(precision_score(y_validate_for_55_64, decision_tree_validate_55_64, zero_division=0),2)
recall_decision_tree_3 = round(recall_score(y_validate_for_55_64, decision_tree_validate_55_64),2)
f1_decision_tree_3 = round(f1_score(y_validate_for_55_64, decision_tree_validate_55_64),2)
roc_decision_tree_3 = round(roc_auc_score(y_validate_for_55_64, decision_tree_validate_55_64),2)

decision_tree.fit(x_train,y_train_for_65_more),
decision_tree_validate_65_more = decision_tree.predict(x_validate)
accuracy_decision_tree_4 = round(accuracy_score(y_validate_for_65_more, decision_tree_validate_65_more),2)
precision_decision_tree_4 = round(precision_score(y_validate_for_65_more, decision_tree_validate_65_more, zero_division=0),2)
recall_decision_tree_4 = round(recall_score(y_validate_for_65_more, decision_tree_validate_65_more),2)
f1_decision_tree_4 = round(f1_score(y_validate_for_65_more, decision_tree_validate_65_more),2)
roc_decision_tree_4 = round(roc_auc_score(y_validate_for_65_more, decision_tree_validate_65_more),2)

#random forest train and validate
random_forest.fit(x_train,y_train_for_35_44)
random_forest_validate_35_44 = random_forest.predict(x_validate)
accuracy_random_forest_1 = round(accuracy_score(y_validate_for_35_44, random_forest_validate_35_44),2)
precision_random_forest_1 = round(precision_score(y_validate_for_35_44, random_forest_validate_35_44, zero_division=0),2)
recall_random_forest_1 = round(recall_score(y_validate_for_35_44, random_forest_validate_35_44),2)
f1_random_forest_1 = round(f1_score(y_validate_for_35_44, random_forest_validate_35_44),2)
roc_random_forest_1 = round(roc_auc_score(y_validate_for_35_44, random_forest_validate_35_44),2)

random_forest.fit(x_train,y_train_for_45_54)
random_forest_validate_45_54 = random_forest.predict(x_validate)
accuracy_random_forest_2 = round(accuracy_score(y_validate_for_45_54, random_forest_validate_45_54),2)
precision_random_forest_2 = round(precision_score(y_validate_for_45_54, random_forest_validate_45_54, zero_division=0),2)
recall_random_forest_2 = round(recall_score(y_validate_for_45_54, random_forest_validate_45_54),2)
f1_random_forest_2 = round(f1_score(y_validate_for_45_54, random_forest_validate_45_54),2)
roc_random_forest_2 = round(roc_auc_score(y_validate_for_45_54, random_forest_validate_45_54),2)

random_forest.fit(x_train,y_train_for_55_64)
random_forest_validate_55_64 = random_forest.predict(x_validate)
accuracy_random_forest_3 = round(accuracy_score(y_validate_for_55_64, random_forest_validate_55_64),2)
precision_random_forest_3 = round(precision_score(y_validate_for_55_64, random_forest_validate_55_64, zero_division=0),2)
recall_random_forest_3 = round(recall_score(y_validate_for_55_64, random_forest_validate_55_64),2)
f1_random_forest_3 = round(f1_score(y_validate_for_55_64, random_forest_validate_55_64),2)
roc_random_forest_3 = round(roc_auc_score(y_validate_for_55_64, random_forest_validate_55_64),2)

random_forest.fit(x_train,y_train_for_65_more)
random_forest_validate_65_more = random_forest.predict(x_validate)
accuracy_random_forest_4 = round(accuracy_score(y_validate_for_65_more, random_forest_validate_65_more),2)
precision_random_forest_4 = round(precision_score(y_validate_for_65_more, random_forest_validate_65_more, zero_division=0),2)
recall_random_forest_4 = round(recall_score(y_validate_for_65_more, random_forest_validate_65_more),2)
f1_random_forest_4 = round(f1_score(y_validate_for_65_more, random_forest_validate_65_more),2)
roc_random_forest_4 = round(roc_auc_score(y_validate_for_65_more, random_forest_validate_65_more),2)

#bgm train and validate
bgm.fit(x_train,y_train_for_35_44)
bgm_validate_35_44 = bgm.predict(x_validate)
accuracy_bgm_1 = round(accuracy_score(y_validate_for_35_44, bgm_validate_35_44),2)
precision_bgm_1 = round(precision_score(y_validate_for_35_44, bgm_validate_35_44, zero_division=0),2)
recall_bgm_1 = round(recall_score(y_validate_for_35_44, bgm_validate_35_44),2)
f1_bgm_1 = round(f1_score(y_validate_for_35_44, bgm_validate_35_44),2)
roc_bgm_1 = round(roc_auc_score(y_validate_for_35_44, bgm_validate_35_44),2)

bgm.fit(x_train,y_train_for_45_54)
bgm_validate_45_54 = bgm.predict(x_validate)
accuracy_bgm_2 = round(accuracy_score(y_validate_for_45_54, bgm_validate_45_54),2)
precision_bgm_2 = round(precision_score(y_validate_for_45_54, bgm_validate_45_54, zero_division=0),2)
recall_bgm_2 = round(recall_score(y_validate_for_45_54, bgm_validate_45_54),2)
f1_bgm_2 = round(f1_score(y_validate_for_45_54, bgm_validate_45_54),2)
roc_bgm_2 = round(roc_auc_score(y_validate_for_45_54, bgm_validate_45_54),2)

bgm.fit(x_train,y_train_for_55_64)
bgm_validate_55_64 = bgm.predict(x_validate)
accuracy_bgm_3 = round(accuracy_score(y_validate_for_55_64, bgm_validate_55_64),2)
precision_bgm_3 = round(precision_score(y_validate_for_55_64, bgm_validate_55_64, zero_division=0),2)
recall_bgm_3 = round(recall_score(y_validate_for_55_64, bgm_validate_55_64),2)
f1_bgm_3 = round(f1_score(y_validate_for_55_64, bgm_validate_55_64),2)
roc_bgm_3 = round(roc_auc_score(y_validate_for_55_64, bgm_validate_55_64),2)

bgm.fit(x_train,y_train_for_65_more)
bgm_validate_65_more = bgm.predict(x_validate)
accuracy_bgm_4 = round(accuracy_score(y_validate_for_65_more, bgm_validate_65_more),2)
precision_bgm_4 = round(precision_score(y_validate_for_65_more, bgm_validate_65_more, zero_division=0),2)
recall_bgm_4 = round(recall_score(y_validate_for_65_more, bgm_validate_65_more),2)
f1_bgm_4 = round(f1_score(y_validate_for_65_more, bgm_validate_65_more),2)
roc_bgm_4 = round(roc_auc_score(y_validate_for_65_more, bgm_validate_65_more),2)

#knn train and validate
k_neighbors.fit(x_train,y_train_for_35_44)
k_neighbors_validate_35_44 = k_neighbors.predict(x_validate)
accuracy_k_neighbors_1 = round(accuracy_score(y_validate_for_35_44, k_neighbors_validate_35_44),2)
precision_k_neighbors_1 = round(precision_score(y_validate_for_35_44, k_neighbors_validate_35_44, zero_division=0),2)
recall_k_neighbors_1 = round(recall_score(y_validate_for_35_44, k_neighbors_validate_35_44),2)
f1_k_neighbors_1 = round(f1_score(y_validate_for_35_44, k_neighbors_validate_35_44),2)
roc_k_neighbors_1 = round(roc_auc_score(y_validate_for_35_44, k_neighbors_validate_35_44),2)

k_neighbors.fit(x_train,y_train_for_45_54)
k_neighbors_validate_45_54 = k_neighbors.predict(x_validate)
accuracy_k_neighbors_2 = round(accuracy_score(y_validate_for_45_54, k_neighbors_validate_45_54),2)
precision_k_neighbors_2 = round(precision_score(y_validate_for_45_54, k_neighbors_validate_45_54, zero_division=0),2)
recall_k_neighbors_2 = round(recall_score(y_validate_for_45_54, k_neighbors_validate_45_54),2)
f1_k_neighbors_2 = round(f1_score(y_validate_for_45_54, k_neighbors_validate_45_54),2)
roc_k_neighbors_2 = round(roc_auc_score(y_validate_for_45_54, k_neighbors_validate_45_54),2)

k_neighbors.fit(x_train,y_train_for_55_64)
k_neighbors_validate_55_64 = k_neighbors.predict(x_validate)
accuracy_k_neighbors_3 = round(accuracy_score(y_validate_for_55_64, k_neighbors_validate_55_64),2)
precision_k_neighbors_3 = round(precision_score(y_validate_for_55_64, k_neighbors_validate_55_64, zero_division=0),2)
recall_k_neighbors_3 = round(recall_score(y_validate_for_55_64, k_neighbors_validate_55_64),2)
f1_k_neighbors_3 = round(f1_score(y_validate_for_55_64, k_neighbors_validate_55_64),2)
roc_k_neighbors_3 = round(roc_auc_score(y_validate_for_55_64, k_neighbors_validate_55_64),2)

k_neighbors.fit(x_train,y_train_for_65_more)
k_neighbors_validate_65_more = k_neighbors.predict(x_validate)
accuracy_k_neighbors_4 = round(accuracy_score(y_validate_for_65_more, k_neighbors_validate_65_more),2)
precision_k_neighbors_4 = round(precision_score(y_validate_for_65_more, k_neighbors_validate_65_more, zero_division=0),2)
recall_k_neighbors_4 = round(recall_score(y_validate_for_65_more, k_neighbors_validate_65_more),2)
f1_k_neighbors_4 = round(f1_score(y_validate_for_65_more, k_neighbors_validate_65_more),2)
roc_k_neighbors_4 = round(roc_auc_score(y_validate_for_65_more, k_neighbors_validate_65_more),2)

#svc train and validate
support_vector.fit(x_train,y_train_for_35_44)
support_vector_validate_35_44 = support_vector.predict(x_validate)
accuracy_support_vector_1 = round(accuracy_score(y_validate_for_35_44, support_vector_validate_35_44),2)
precision_support_vector_1 = round(precision_score(y_validate_for_35_44, support_vector_validate_35_44, zero_division=0),2)
recall_support_vector_1 = round(recall_score(y_validate_for_35_44, support_vector_validate_35_44),2)
f1_support_vector_1 = round(f1_score(y_validate_for_35_44, support_vector_validate_35_44),2)
roc_support_vector_1 = round(roc_auc_score(y_validate_for_35_44, support_vector_validate_35_44),2)

support_vector.fit(x_train,y_train_for_45_54)
support_vector_validate_45_54 = support_vector.predict(x_validate)
accuracy_support_vector_2 = round(accuracy_score(y_validate_for_45_54, support_vector_validate_45_54),2)
precision_support_vector_2 = round(precision_score(y_validate_for_45_54, support_vector_validate_45_54, zero_division=0),2)
recall_support_vector_2 = round(recall_score(y_validate_for_45_54, support_vector_validate_45_54),2)
f1_support_vector_2 = round(f1_score(y_validate_for_45_54, support_vector_validate_45_54),2)
roc_support_vector_2 = round(roc_auc_score(y_validate_for_45_54, support_vector_validate_45_54),2)

support_vector.fit(x_train,y_train_for_55_64)
support_vector_validate_55_64 = support_vector.predict(x_validate)
accuracy_support_vector_3 = round(accuracy_score(y_validate_for_55_64, support_vector_validate_55_64),2)
precision_support_vector_3 = round(precision_score(y_validate_for_55_64, support_vector_validate_55_64, zero_division=0),2)
recall_support_vector_3 = round(recall_score(y_validate_for_55_64, support_vector_validate_55_64),2)
f1_support_vector_3 = round(f1_score(y_validate_for_55_64, support_vector_validate_55_64),2)
roc_support_vector_3 = round(roc_auc_score(y_validate_for_55_64, support_vector_validate_55_64),2)

support_vector.fit(x_train,y_train_for_65_more)
support_vector_validate_65_more = support_vector.predict(x_validate)
accuracy_support_vector_4 = round(accuracy_score(y_validate_for_65_more, support_vector_validate_65_more),2)
precision_support_vector_4 = round(precision_score(y_validate_for_65_more, support_vector_validate_65_more, zero_division=0),2)
recall_support_vector_4 = round(recall_score(y_validate_for_65_more, support_vector_validate_65_more),2)
f1_support_vector_4 = round(f1_score(y_validate_for_65_more, support_vector_validate_65_more),2)
roc_support_vector_4 = round(roc_auc_score(y_validate_for_65_more, support_vector_validate_65_more),2)




